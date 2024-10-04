import tkinter as tk
from tkinter import ttk
import threading
from typing import Tuple, Optional

from pynput import mouse

# Import necessary modules for screenshot and image processing
from PIL import ImageGrab

from img2sgf import sgf_from_image

Pixel = tuple[int, int]

# Known for my screen
# REGION = (368, 255, 1029, 926)
REGION = None


class App:
    def __init__(self, root: tk.Tk, interval: int) -> None:
        """
        Initialize the application.

        Args:
            root: The main Tkinter window.
            interval: The default interval in milliseconds between screenshots.
        """
        self.root = root
        self.interval = interval  # Default interval in milliseconds
        # self.region: Optional[Tuple[int, int, int, int]] = None  # Screenshot region
        self.region: Optional[Tuple[int, int, int, int]] = REGION
        self.running = False  # Flag to control the periodic task
        self.task_started = False  # Flag to prevent multiple starts
        self.setup_ui()
        self.root.protocol(
            "WM_DELETE_WINDOW", self.on_closing
        )  # Handle window close event

    def setup_ui(self) -> None:
        """
        Set up the user interface.
        """
        self.root.title("Go Game Screen Capture")

        # Create frames for layout
        input_frame = ttk.Frame(self.root)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Interval input
        interval_label = ttk.Label(input_frame, text="Interval (ms):")
        interval_label.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.interval_entry = ttk.Entry(input_frame, width=20)
        self.interval_entry.grid(row=0, column=1, pady=5)
        self.interval_entry.insert(0, str(self.interval))  # Default value

        # SGF file input
        sgf_label = ttk.Label(input_frame, text="SGF File:")
        sgf_label.grid(row=1, column=0, sticky=tk.W, pady=5)

        self.sgf_entry = ttk.Entry(input_frame, width=50)
        self.sgf_entry.grid(row=1, column=1, pady=5)
        self.sgf_entry.insert(0, "game.sgf")  # Placeholder value

        # Screenshot file input
        file_label = ttk.Label(input_frame, text="Screenshots file:")
        file_label.grid(row=2, column=0, sticky=tk.W, pady=5)

        self.file_entry = ttk.Entry(input_frame, width=50)
        self.file_entry.grid(row=2, column=1, pady=5)
        self.file_entry.insert(0, "screenshot.png")  # Default file

        # Select Region button
        self.select_region_button = ttk.Button(
            input_frame, text="Select Region", command=self.select_region
        )
        self.select_region_button.grid(row=3, column=0, pady=10)

        # Start and Stop buttons
        self.start_button = ttk.Button(
            input_frame, text="Start", command=self.start_periodic_task
        )
        self.start_button.grid(row=3, column=1, pady=10, sticky=tk.W)

        self.stop_button = ttk.Button(
            input_frame, text="Stop", command=self.stop_periodic_task
        )
        self.stop_button.grid(row=3, column=1, pady=10, sticky=tk.E)
        self.stop_button.config(state=tk.DISABLED)  # Initially disabled

    def select_region(self) -> None:
        """
        Allows the user to select a region of the screen.
        """
        board_assigned = BoardAssigner()
        one, two = board_assigned.get_left_top_and_right_bottom_chessboard_pixels()
        self.region = (one[0], one[1], two[0], two[1])
        print(f"Selected region: {self.region}")

    def start_periodic_task(self) -> None:
        """
        Start the periodic screenshot capture.
        """
        if not self.task_started:
            if self.region is None:
                print("Please select a region first.")
                return

            interval_str = self.interval_entry.get()
            if not interval_str.isdigit():
                print("Interval must be a positive integer.")
                return
            self.interval = int(interval_str)

            self.sgf_file = self.sgf_entry.get()
            if not self.sgf_file:
                print("Please specify an SGF file.")
                return

            self.screenshot_file = self.file_entry.get()
            if not self.screenshot_file:
                print("Please specify a screenshot file.")
                return

            self.running = True
            self.task_started = True
            # Disable the Start button and inputs
            self.start_button.config(state=tk.DISABLED)
            self.select_region_button.config(state=tk.DISABLED)
            self.interval_entry.config(state=tk.DISABLED)
            self.sgf_entry.config(state=tk.DISABLED)
            self.file_entry.config(state=tk.DISABLED)
            # Enable the Stop button
            self.stop_button.config(state=tk.NORMAL)

            # Start the periodic task
            self.root.after(0, self.periodic_task)
            print("Periodic task started.")

    def stop_periodic_task(self) -> None:
        """
        Stop the periodic screenshot capture.
        """
        if self.running:
            self.running = False
            self.task_started = False
            # Enable the Start button and inputs
            self.start_button.config(state=tk.NORMAL)
            self.select_region_button.config(state=tk.NORMAL)
            self.interval_entry.config(state=tk.NORMAL)
            self.sgf_entry.config(state=tk.NORMAL)
            self.file_entry.config(state=tk.NORMAL)
            # Disable the Stop button
            self.stop_button.config(state=tk.DISABLED)
            print("Periodic task stopped.")

    def periodic_task(self) -> None:
        """
        Capture screenshot and process it, then reschedule itself.
        """
        if self.running:
            # Capture and process the screenshot in a separate thread
            threading.Thread(target=self.capture_and_process).start()
            # Schedule the next capture
            self.root.after(self.interval, self.periodic_task)

    def capture_and_process(self) -> None:
        """
        Capture the screenshot of the selected region and process it.
        """
        if self.region is None:
            return

        # Capture the screenshot
        screenshot = ImageGrab.grab(bbox=self.region)
        print("Screenshot captured.")

        screenshot.save(self.screenshot_file)

        # Placeholder for custom processing logic
        # For example, analyze the screenshot and output to SGF file
        sgf_data = self.process_screenshot(self.screenshot_file)

        # Save the SGF data to the file
        try:
            with open(self.sgf_file, "w") as sgf_file:
                sgf_file.write(sgf_data)
            print(f"SGF data saved to {self.sgf_file}")
        except Exception as e:
            print(f"Error saving SGF file: {e}")

    def process_screenshot(self, file: str) -> str:
        """
        Placeholder function to process the screenshot and return SGF data.

        Args:
            image: The captured screenshot image.

        Returns:
            A string containing the SGF data.
        """
        # sgf_data = "(;GM[1]FF[4]CA[UTF-8]SZ[19];B[pd];W[dd])"
        sgf_data = sgf_from_image(file)
        print("sgf_data", sgf_data)
        return sgf_data

    def on_closing(self) -> None:
        """
        Handle the window close event.
        """
        self.running = False
        self.root.destroy()


class BoardAssigner:
    def __init__(self) -> None:
        self._left_top: "Pixel" | None = None
        self._right_bottom: "Pixel" | None = None

    def get_left_top_and_right_bottom_chessboard_pixels(
        self,
    ) -> tuple["Pixel", "Pixel"]:
        """Returns boundaries of the chessboard"""
        print("Please rightlick the most upperleft corner of the chessboard")
        with mouse.Listener(on_click=self._assign_two_corners_on_click) as listener:
            listener.join()

        print("Boundaries assigned, you may want to save them into config.py")

        assert self._left_top is not None
        assert self._right_bottom is not None
        return self._left_top, self._right_bottom

    def _assign_two_corners_on_click(
        self, x: int, y: int, button, pressed: bool
    ) -> bool:
        """Listen for right-clicks and assign the two corners"""
        if button == mouse.Button.right and pressed:
            if self._left_top is None:
                self._left_top = (x, y)
                print(f"chessboard_left_top_pixel assigned - {x},{y}")
                print("Please rightlick the most bottomright corner of the chessboard")
            elif self._right_bottom is None:
                self._right_bottom = (x, y)
                print(f"chessboard_right_bottom_pixel assigned - {x},{y}")

        return not self._stop_listening_on_mouse_input()

    def _stop_listening_on_mouse_input(self) -> bool:
        """Stop whenever both corners are assigned"""
        if self._left_top is not None and self._right_bottom is not None:
            print("Stopping the assignment")
            return True
        return False


def main() -> None:
    """
    The main entry point of the application.
    """
    root = tk.Tk()
    interval = 2000
    app = App(root, interval)

    root.mainloop()


if __name__ == "__main__":
    main()
