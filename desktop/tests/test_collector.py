import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mousewatch.collector.buffer import MovementAssembler, direction_angle
from mousewatch.collector.listener import button_changes
from mousewatch.storage.traces import HEADER, TraceWriter, format_record


class DirectionAngleTests(unittest.TestCase):
    def test_horizontal_and_vertical(self) -> None:
        self.assertEqual(direction_angle(10, 0), 0)
        self.assertEqual(direction_angle(-10, 0), 0)
        self.assertEqual(direction_angle(0, 10), 90)
        self.assertEqual(direction_angle(0, -10), -90)

    def test_slope_stays_inside_dataset_range(self) -> None:
        self.assertEqual(direction_angle(10, 10), 45)
        self.assertEqual(direction_angle(-10, 10), -45)
        self.assertEqual(direction_angle(-10, -10), 45)
        self.assertEqual(direction_angle(5, -5), -45)
        for dx in range(-20, 21):
            for dy in range(-20, 21):
                angle = direction_angle(dx, dy)
                self.assertGreaterEqual(angle, -90)
                self.assertLessEqual(angle, 90)


class AssemblerTests(unittest.TestCase):
    def _assembler(self) -> MovementAssembler:
        return MovementAssembler(user_id=5907, pause_ns=100_000_000, min_distance_px=4)

    def test_straight_movement_keeps_its_own_direction(self) -> None:
        assembler = self._assembler()
        self.assertEqual(assembler.add_move(6, 0, 0), [])
        self.assertEqual(assembler.add_move(6, 1, 30_000_000), [])
        records = assembler.flush(200_000_000)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsNone(record.duration_ms)
        self.assertEqual(record.angle, 5)
        self.assertEqual(record.distance_px, 12)
        self.assertEqual(record.velocity_px_s, 400)
        self.assertEqual(record.when_ns, 30_000_000)

    def test_turn_inside_a_slice_stays_one_row(self) -> None:
        assembler = self._assembler()
        self.assertEqual(assembler.add_move(10, 0, 0), [])
        self.assertEqual(assembler.add_move(0, 10, 20_000_000), [])
        records = assembler.flush(200_000_000)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].angle, 45)
        self.assertEqual(records[0].distance_px, 14)

    def test_continuous_stroke_writes_about_eight_rows_a_second(self) -> None:
        assembler = MovementAssembler(
            5907, pause_ns=100_000_000, segment_ns=130_000_000
        )
        rows = []
        for step in range(41):
            rows.extend(assembler.add_move(5, 0, step * 10_000_000))
        self.assertEqual(len(rows), 3)
        self.assertEqual([record.distance_px for record in rows], [65, 65, 65])
        self.assertTrue(all(record.angle == 0 for record in rows))
        gaps = [rows[i].when_ns - rows[i - 1].when_ns for i in range(1, 3)]
        self.assertEqual(gaps, [130_000_000, 130_000_000])

    def test_rapid_clicks_are_all_kept(self) -> None:
        assembler = self._assembler()
        rows = []
        for index in range(20):
            down = index * 50_000_000
            rows.extend(assembler.add_button("left", True, down))
            rows.extend(assembler.add_button("left", False, down + 10_000_000))
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(record.duration_ms == 10 for record in rows))
        self.assertTrue(all(record.angle is None for record in rows))

    def test_short_movement_is_dropped(self) -> None:
        assembler = self._assembler()
        assembler.add_move(1, 0, 0)
        assembler.add_move(1, 0, 10_000_000)
        self.assertEqual(assembler.flush(200_000_000), [])

    def test_pause_starts_a_new_movement(self) -> None:
        assembler = self._assembler()
        self.assertEqual(assembler.add_move(10, 0, 0), [])
        closed = assembler.add_move(10, 0, 200_000_000)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].distance_px, 10)
        self.assertEqual(closed[0].angle, 0)
        second = assembler.flush(400_000_000)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].distance_px, 10)
        self.assertEqual(second[0].angle, 0)

    def test_click_duration_and_null_movement_fields(self) -> None:
        assembler = self._assembler()
        self.assertEqual(assembler.add_button("left", True, 1_000_000_000), [])
        records = assembler.add_button("left", False, 1_048_000_000)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.duration_ms, 48)
        self.assertIsNone(record.angle)
        self.assertIsNone(record.distance_px)
        self.assertIsNone(record.velocity_px_s)

    def test_press_closes_the_movement_first(self) -> None:
        assembler = self._assembler()
        assembler.add_move(10, 0, 0)
        closed = assembler.add_button("left", True, 30_000_000)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].distance_px, 10)
        self.assertEqual(closed[0].angle, 0)
        self.assertIsNone(closed[0].duration_ms)
        click = assembler.add_button("left", False, 78_000_000)
        self.assertEqual(click[0].duration_ms, 48)


class FormatTests(unittest.TestCase):
    def test_movement_and_click_rows(self) -> None:
        assembler = MovementAssembler(5907, pause_ns=100_000_000)
        started = 1_700_000_000_000_000_000
        assembler.add_move(856, 0, started)
        assembler.add_move(856, 0, started + 50_000_000)
        movement = assembler.flush(started + 200_000_000)[0]
        self.assertEqual(
            assembler.add_button("left", False, 1_700_000_000_600_000_000),
            [],
        )
        assembler.add_button("left", True, 1_700_000_000_552_000_000)
        click_row = assembler.add_button("left", False, 1_700_000_000_600_000_000)[0]

        movement_line = format_record(movement).rstrip("\n")
        user, duration, angle, distance, velocity, date, clock = movement_line.split(";")
        self.assertEqual(
            [user, duration, angle, distance, velocity],
            ["5907", "NULL", "0", "1712", "34240"],
        )
        self.assertRegex(date, r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(clock, r"^\d{2}:\d{2}:\d{2}\.\d{7}$")

        click_line = format_record(click_row).rstrip("\n")
        self.assertRegex(
            click_line,
            r"^5907;48;NULL;NULL;NULL;\d{4}-\d{2}-\d{2};\d{2}:\d{2}:\d{2}\.\d{7}$",
        )

    def test_writer_appends_without_a_second_header(self) -> None:
        assembler = MovementAssembler(1, pause_ns=1)
        assembler.add_move(5, 0, 0)
        record = assembler.flush(2_000_000)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.csv"
            with TraceWriter(path, flush_every=1) as writer:
                writer.write(record)
            with TraceWriter(path, flush_every=1) as writer:
                writer.write(record)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], HEADER)
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[1].startswith("1;NULL;0;5;"))
        self.assertTrue(lines[2].startswith("1;NULL;0;5;"))


class ButtonFlagTests(unittest.TestCase):
    def test_clicks_ignore_the_wheel(self) -> None:
        self.assertEqual(button_changes(0x0001), [("left", True)])
        self.assertEqual(button_changes(0x0002), [("left", False)])
        self.assertEqual(button_changes(0x0001 | 0x0002), [("left", True), ("left", False)])
        self.assertEqual(button_changes(0x0400), [])
        self.assertEqual(button_changes(0x0004 | 0x0400), [("right", True)])


if __name__ == "__main__":
    unittest.main()
