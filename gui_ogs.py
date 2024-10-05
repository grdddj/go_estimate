import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import requests
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

INTERVAL = 2000

TITLE = "OGS Game Monitor (by grdddj)"


class App:
    def __init__(self, root: tk.Tk, interval: int) -> None:
        """
        Initialize the application.

        Args:
            root: The main Tkinter window.
            interval: The default interval in milliseconds between data updates.
        """
        self.root = root
        self.interval = interval  # Default interval in milliseconds
        self.running = False  # Flag to control the periodic task
        self.task_started = False  # Flag to prevent multiple starts
        self.values = []

        self.game_id = None
        self.backend_url = None
        self.last_move_number = 0  # To keep track of moves

        self.setup_ui()
        self.root.protocol(
            "WM_DELETE_WINDOW", self.on_closing
        )  # Handle window close event

    def setup_ui(self) -> None:
        """
        Set up the user interface.
        """
        self.root.title(TITLE)

        # Create frames for layout
        input_frame = ttk.Frame(self.root)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Game ID input
        game_id_label = ttk.Label(input_frame, text="Game ID:")
        game_id_label.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.game_id_entry = ttk.Entry(input_frame, width=20)
        self.game_id_entry.grid(row=0, column=1, pady=5)
        self.game_id_entry.insert(0, "68382900")  # Placeholder value

        # Backend URL input
        backend_label = ttk.Label(input_frame, text="Backend URL:")
        backend_label.grid(row=1, column=0, sticky=tk.W, pady=5)

        self.backend_entry = ttk.Entry(input_frame, width=50)
        self.backend_entry.grid(row=1, column=1, pady=5)
        self.backend_entry.insert(0, "http://localhost:8590/sgf")  # Placeholder value

        # Start and Stop buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(
            button_frame, text="Start", command=self.start_task
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_task)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)  # Initially disabled

        # Player info labels
        self.player_info_label = ttk.Label(input_frame, text="")
        self.player_info_label.grid(row=3, column=0, columnspan=2, pady=5)

        # Matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        (self.line,) = self.ax.plot([], [], "b-")

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def start_task(self) -> None:
        """
        Start the OGS game monitoring task.
        """
        if not self.task_started:
            game_id_str = self.game_id_entry.get()
            if not game_id_str.isdigit():
                print("Game ID must be a number.")
                return
            self.game_id = int(game_id_str)

            self.backend_url = self.backend_entry.get()
            if not self.backend_url:
                print("Please specify the backend URL.")
                return

            # Fetch game information
            game_info = self.fetch_game_info()
            if game_info is None:
                print("Failed to fetch game information.")
                return

            # Display player information
            self.display_player_info(game_info)

            # Initialize the last move number
            self.last_move_number = self.get_move_number(game_info)

            self.running = True
            self.task_started = True

            # Disable the Start button and inputs
            self.start_button.config(state=tk.DISABLED)
            self.game_id_entry.config(state=tk.DISABLED)
            self.backend_entry.config(state=tk.DISABLED)
            # Enable the Stop button
            self.stop_button.config(state=tk.NORMAL)

            # Clear previous data
            self.values = []
            self.update_plot(clear=True)

            # Start the periodic task
            threading.Thread(target=self.periodic_task, daemon=True).start()
            print("Monitoring started.")

    def stop_task(self) -> None:
        """
        Stop the OGS game monitoring task.
        """
        self.running = False
        self.task_started = False

        # Enable the Start button and inputs
        self.start_button.config(state=tk.NORMAL)
        self.game_id_entry.config(state=tk.NORMAL)
        self.backend_entry.config(state=tk.NORMAL)
        # Disable the Stop button
        self.stop_button.config(state=tk.DISABLED)
        print("Monitoring stopped.")

    def on_closing(self) -> None:
        """
        Handle the window close event.
        """
        self.stop_task()
        self.root.destroy()

    def fetch_game_info(self) -> Optional[dict]:
        """
        Fetches game information from OGS API.

        Returns:
            A dictionary containing game information, or None if failed.
        """
        try:
            url = f"https://online-go.com/api/v1/games/{self.game_id}"
            response = requests.get(url)
            if response.status_code == 200:
                game_info = response.json()
                return game_info
            else:
                print(f"Failed to fetch game info. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching game info: {e}")
            return None

    def display_player_info(self, game_info: dict) -> None:
        """
        Displays player information in the GUI.

        Args:
            game_info: The game information dictionary.
        """
        try:
            black_player = game_info["players"]["black"]["username"]
            white_player = game_info["players"]["white"]["username"]
            self.player_info_label.config(
                text=f"Black: {black_player} vs White: {white_player}"
            )
        except Exception as e:
            print(f"Error displaying player info: {e}")

    def get_move_number(self, game_info: dict) -> int:
        """
        Extracts the current move number from game information.

        Args:
            game_info: The game information dictionary.

        Returns:
            The current move number as an integer.
        """
        try:
            moves = game_info.get("gamedata", {}).get("moves", [])
            return len(moves)
        except Exception as e:
            print(f"Error getting move number: {e}")
            return 0

    def periodic_task(self) -> None:
        """
        Periodically checks for new moves.
        """
        while self.running:
            game_info = self.fetch_game_info()
            if game_info is not None:
                phase = game_info["gamedata"].get("phase")
                if phase == "finished":
                    print("Game has finished.")
                    self.running = False
                    # Update the GUI in the main thread
                    self.root.after(0, self.on_game_over)
                    break  # Exit the loop

                current_move_number = self.get_move_number(game_info)
                if current_move_number > self.last_move_number:
                    print(f"New move detected. Move number: {current_move_number}")
                    self.last_move_number = current_move_number
                    sgf_content = self.build_sgf(game_info)
                    # Send SGF content to the backend
                    self.send_sgf_to_backend(sgf_content)
            time.sleep(self.interval / 1000.0)  # Convert milliseconds to seconds

    def build_sgf(self, game_info: dict) -> str:
        """
        Builds the SGF content from game information.

        Args:
            game_info: The game information dictionary.

        Returns:
            The SGF content as a string.
        """
        try:
            # SGF header
            sgf = "(;GM[1]"
            # Board size
            board_size = game_info.get("width", 19)
            sgf += f"SZ[{board_size}]"
            # Komi
            komi = game_info.get("komi", 0)
            sgf += f"KM[{komi}]"
            # Players
            black_player = game_info["players"]["black"]["username"]
            white_player = game_info["players"]["white"]["username"]
            sgf += f"PB[{black_player}]PW[{white_player}]"
            # Moves
            moves = game_info["gamedata"].get("moves", [])
            for index, move in enumerate(moves):
                x, y = move[0], move[1]
                if x is None or y is None:
                    continue  # Pass move
                # Convert coordinates to SGF format (a-s)
                sgf_x = chr(ord("a") + x)
                sgf_y = chr(ord("a") + y)
                player = "B" if index % 2 == 0 else "W"
                sgf += f";{player}[{sgf_x}{sgf_y}]"
            sgf += ")"
            return sgf
        except Exception as e:
            print(f"Error building SGF: {e}")
            return ""

    def send_sgf_to_backend(self, sgf_content: str) -> None:
        """
        Sends the SGF content to the backend and updates the graph based on the response.

        Args:
            sgf_content: The SGF file content as a string.
        """
        try:
            assert self.backend_url
            response = requests.post(
                self.backend_url, json={"sgf": sgf_content, "visits": 20}
            )
            if response.status_code == 200:
                json_response = response.json()
                score_lead = json_response.get("scoreLead")
                if isinstance(score_lead, (int, float)):
                    self.values.append(score_lead)
                    print(f"Received Score Lead: {score_lead}")
                    # Update plot on the main thread
                    self.root.after(0, self.update_plot)
                else:
                    print(f"Invalid value in response - {json_response}")
            else:
                print(f"Backend request failed. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending SGF to backend: {e}")

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

    def on_game_over(self) -> None:
        """
        Handles the game over event by stopping the task and updating the GUI.
        """
        self.stop_task()
        # Display a message to the user
        print("Monitoring stopped. The game has ended.")
        # Optionally, display a message box
        messagebox.showinfo("Game Over", "The game has finished. Monitoring stopped.")


def main() -> None:
    """
    The main entry point of the application.
    """
    root = tk.Tk()
    App(root, INTERVAL)

    root.mainloop()


if __name__ == "__main__":
    main()
