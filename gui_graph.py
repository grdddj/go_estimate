import asyncio
import datetime
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import aiohttp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

TITLE = "Go estimation graph"
INTERVAL_MS = 2000


class App:
    def __init__(
        self, root: tk.Tk, loop: asyncio.AbstractEventLoop, interval: int
    ) -> None:
        """
        Initialize the application.

        Args:
            root: The main Tkinter window.
            loop: The asyncio event loop.
            interval: The interval in milliseconds between measurements.
        """
        self.root = root
        self.loop = loop
        self.interval = interval  # in milliseconds
        self.values: list[float] = []
        self.measurement_number: int = 0
        self.running = False  # Flag to control the periodic task
        self.task_started = False  # Flag to prevent multiple starts

        self.setup_ui()
        self.root.protocol(
            "WM_DELETE_WINDOW", self.on_closing
        )  # Handle window close event

    def setup_ui(self) -> None:
        """
        Set up the user interface, including the matplotlib figure embedded in Tkinter.
        """
        self.root.title(TITLE)

        # Create frames for layout
        input_frame = ttk.Frame(self.root)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File path input
        file_label = ttk.Label(input_frame, text="File Path:")
        file_label.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.file_entry = ttk.Entry(input_frame, width=50)
        self.file_entry.grid(row=0, column=1, pady=5)
        self.file_entry.insert(0, "game.sgf")  # Placeholder value

        # Server URL input
        url_label = ttk.Label(input_frame, text="Server URL:")
        url_label.grid(row=1, column=0, sticky=tk.W, pady=5)

        self.url_entry = ttk.Entry(input_frame, width=50)
        self.url_entry.grid(row=1, column=1, pady=5)
        self.url_entry.insert(0, "http://localhost:8590/sgf")  # Placeholder value

        # Start and Stop buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(
            button_frame, text="Start", command=self.start_periodic_task
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            button_frame, text="Stop", command=self.stop_periodic_task
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)  # Initially disabled

        self.download_button = ttk.Button(
            button_frame, text="Download", command=self.download_plot
        )
        self.download_button.pack(side=tk.LEFT, padx=5)

        # Matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        (self.line,) = self.ax.plot([], [], "b-")

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def start_periodic_task(self) -> None:
        """
        Start the periodic task when the Start button is pressed.
        """
        if not self.task_started:
            self.file_path = self.file_entry.get()
            self.server_url = self.url_entry.get()

            if not self.file_path or not self.server_url:
                print("File path and Server URL must not be empty.")
                return

            if not Path(self.file_path).is_file():
                print("File does not exist.")
                return

            # You can add more validation here if needed

            self.running = True
            self.task_started = True
            # Disable the Start button and text inputs
            self.start_button.config(state=tk.DISABLED)
            self.file_entry.config(state=tk.DISABLED)
            self.url_entry.config(state=tk.DISABLED)
            # Enable the Stop button
            self.stop_button.config(state=tk.NORMAL)

            # Reset values and update plot
            self.values = []
            self.measurement_number = 0
            self.root.after(0, self.update_plot, True)

            # Start the periodic task
            self.root.after(0, self.periodic_task)
            print("Periodic task started.")

    def stop_periodic_task(self) -> None:
        """
        Stop the periodic task when the Stop button is pressed.
        """
        if self.running:
            self.running = False  # Stop the periodic task from rescheduling
            self.task_started = False  # Allow task to be started again
            # Enable the Start button and text inputs
            self.start_button.config(state=tk.NORMAL)
            self.file_entry.config(state=tk.NORMAL)
            self.url_entry.config(state=tk.NORMAL)
            # Disable the Stop button
            self.stop_button.config(state=tk.DISABLED)
            print("Periodic task stopped.")

    def download_plot(self) -> None:
        """
        Save the current plot to a file with a unique name.
        """
        try:
            # Generate a unique filename using the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plot_{timestamp}.png"
            self.figure.savefig(filename)
            print(f"Plot saved as {filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    def on_closing(self) -> None:
        """
        Handle the window close event by stopping the periodic task and closing the application.
        """
        self.running = False  # Stop the periodic task from rescheduling
        self.root.destroy()  # Close the Tkinter window

    def periodic_task(self) -> None:
        """
        Schedule the asynchronous file reading and sending task, then reschedule itself.
        """
        if self.running:
            asyncio.run_coroutine_threadsafe(self.read_file_and_send(), self.loop)
            self.root.after(self.interval, self.periodic_task)

    async def read_file_and_send(self) -> None:
        """
        Asynchronously read the file content, send it to the server, and process the response.
        """
        # Read the content of the file
        try:
            with open(self.file_path, "r") as file:
                content: str = file.read()
            print(f"Read content from {self.file_path}")
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        # Send the content to the server URL
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.server_url, json={"sgf": content, "visits": 20}
                ) as response:
                    if response.status == 200:
                        json_response = await response.json()
                        print("json_response", json_response)
                        score_lead = json_response.get("scoreLead")
                        if isinstance(score_lead, (int, float)):
                            self.measurement_number += 1
                            self.values.append(score_lead)
                            print(f"Received value: {score_lead}")
                            # Update plot on the main thread
                            self.root.after(0, self.update_plot)
                        else:
                            print("Invalid value in response")
                    else:
                        print(f"HTTP Error: {response.status}")
        except Exception as e:
            print(f"Error sending request: {e}")

    def update_plot(self, clear: bool = False) -> None:
        """
        Update the matplotlib plot with the new data.

        Args:
            clear: If True, clear the plot.
        """
        if clear:
            self.ax.clear()
            (self.line,) = self.ax.plot([], [], "b-")
            self.canvas.draw()
            return

        x_values = list(range(1, len(self.values) + 1))
        y_values = self.values

        self.line.set_data(x_values, y_values)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


def start_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Start the asyncio event loop in a new thread.

    Args:
        loop: The asyncio event loop to run.
    """
    asyncio.set_event_loop(loop)
    loop.run_forever()


def main() -> None:
    """
    The main entry point of the application.
    """
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_loop, args=(loop,))
    t.start()

    root = tk.Tk()
    App(root, loop, interval=INTERVAL_MS)

    try:
        root.mainloop()
    finally:
        # When the Tkinter main loop exits, stop the asyncio loop and wait for the thread to finish
        loop.call_soon_threadsafe(loop.stop)
        t.join()


if __name__ == "__main__":
    main()
