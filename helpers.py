import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, TypedDict

import requests  # type: ignore
from sgfmill import sgf, sgf_moves  # type: ignore

HERE = Path(__file__).parent

CONFIG_DIR = HERE / "configs"
ANALYSIS_CONFIG = CONFIG_DIR / "analysis.cfg"
EVALSGF_CONFIG = CONFIG_DIR / "evalsgf.cfg"

KATAGO = HERE / "katago"
KATAGO_MODEL_BIN = HERE / "katago.gz"


@dataclass
class MoveInfo:
    move: str
    score: float


@dataclass
class PositionInfo:
    scoreLead: float
    moveInfos: list[MoveInfo]


class GameInputData(TypedDict):
    initial_stones: list[list[str]]
    moves: list[list[str]]
    rules: str
    komi: float
    max_visits: int


def get_json_from_output(output: str) -> dict | None:
    for line in output.split("\n"):
        if line.startswith("{"):
            return json.loads(line)
    return None


def get_score_from_output(json_output: dict) -> float:
    return json_output["rootInfo"]["scoreLead"]


def get_move_infos_from_output(json_output: dict) -> list[MoveInfo]:
    move_infos = []
    for move in json_output["moveInfos"]:
        move_infos.append(MoveInfo(move=move["move"], score=move["scoreLead"]))
    return move_infos


def get_position_info_from_output(output: str) -> PositionInfo | None:
    json_output = get_json_from_output(output)
    return get_position_info_from_json_output(json_output)


def get_position_info_from_json_output(json_output: dict | None) -> PositionInfo | None:
    if not json_output:
        print("Failed to parse JSON output")
        return None

    move_infos = get_move_infos_from_output(json_output)
    score = get_score_from_output(json_output)
    return PositionInfo(scoreLead=score, moveInfos=move_infos)


def download_ogs_game(game_id: int) -> str | None:
    url = f"https://online-go.com/api/v1/games/{game_id}/sgf"
    response = requests.get(url)
    if response.status_code == 200:
        sgf_file = f"{game_id}.sgf"
        with open(sgf_file, "wb") as f:
            f.write(response.content)
        return sgf_file
    else:
        print(f"Failed to download game {game_id}")
        return None


def get_number_of_moves(file: str) -> int:
    with open(file, "r") as f:
        sgf_content = f.read()

    sgf_game = sgf.Sgf_game.from_string(sgf_content)

    move_count = 0
    main_sequence = sgf_game.get_main_sequence()
    for node in main_sequence:
        move = node.get_move()
        if move[0] is not None:
            move_count += 1

    return move_count


def sgf_to_data(sgf_content: str, visits: int) -> GameInputData:
    """
    Parses SGF content and extracts initial stones, moves, rules, komi, and max_visits.

    Args:
        sgf_content: The content of the SGF file as a string.

    Returns:
        A dictionary containing the game data in the specified format.
    """
    sgf_game = sgf.Sgf_game.from_string(sgf_content)
    root_node = sgf_game.get_root()

    # Extract rules
    try:
        rules = root_node.get("RU")
    except KeyError as e:
        print(f"Error getting rules: {e}")
        rules = None
    if rules is None:
        rules = "japanese"

    # Extract komi
    try:
        komi_str = root_node.get("KM")
    except KeyError as e:
        print(f"Error getting komi: {e}")
        komi_str = None
    if komi_str is not None:
        komi = float(komi_str)
    else:
        komi = 6.5  # Default to 6.5 if not specified

    # Extract initial stones and moves
    board_size = sgf_game.get_size()

    initial_stones: list[list[str]] = []
    moves: list[list[str]] = []

    # Use sgf_moves to extract setup stones and moves
    try:
        init_board, plays = sgf_moves.get_setup_and_moves(sgf_game)
    except ValueError as e:
        # Handle invalid SGF files
        raise ValueError(f"Invalid SGF file: {e}")

    # Extract initial stones from init_board
    for row in range(board_size):
        for col in range(board_size):
            color = init_board.get(row, col)
            if color is not None:
                point = coords_to_gtp_point((row, col), board_size)
                initial_stones.append([color.upper(), point])

    # Extract moves from plays
    for color, move in plays:
        if move is None:
            continue  # Pass move or no move
        point = coords_to_gtp_point(move, board_size)
        moves.append([color.upper(), point])

    return {
        "initial_stones": initial_stones,
        "moves": moves,
        "rules": rules,
        "komi": komi,
        "max_visits": visits,
    }


def coords_to_gtp_point(coords: Tuple[int, int], board_size: int) -> str:
    """
    Converts SGF coordinates to GTP point notation.

    Args:
        coords: A tuple (row, col) representing the board coordinates.
        board_size: The size of the Go board (e.g., 19).

    Returns:
        A string representing the point in GTP notation (e.g., 'D4').
    """
    if coords is None:
        return ""  # Handle pass moves
    row, col = coords
    # GTP columns: 'A'..'T' excluding 'I'
    col_letters = []
    for c in range(board_size):
        letter = chr(ord("A") + c)
        if letter >= "I":
            letter = chr(ord(letter) + 1)  # Skip 'I'
        col_letters.append(letter)
    col_letter = col_letters[col]
    # GTP rows: 1 to board_size, from bottom to top
    row_number = str(board_size - row)
    return f"{col_letter}{row_number}"


def get_position_info_evalsgf(
    sgf_file: str, move: int, visits: int
) -> PositionInfo | None:
    try:
        command: list[str] = [
            str(KATAGO),
            "evalsgf",
            "-model",
            str(KATAGO_MODEL_BIN),
            "-config",
            str(EVALSGF_CONFIG),
            "-move-num",
            str(move),
            "-v",
            str(visits),
            sgf_file,
            "-print-json",
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        return get_position_info_from_output(output)

    except subprocess.CalledProcessError as e:
        print("Error running KataGo:", e.stderr)
        return None
    except json.JSONDecodeError:
        print("Failed to parse JSON output")
        return None


def _example_sgf_to_data():
    sgf_file = HERE / "game.sgf"
    with open(sgf_file, "r") as f:
        sgf_content = f.read()
    game_data = sgf_to_data(sgf_content, 50)
    print("game_data", game_data)


def _example_analyze_ogs():
    game_id = 68285929

    sgf_file = download_ogs_game(game_id)

    if not sgf_file:
        print("Failed to download game")
        exit(1)

    visits = 20

    move = get_number_of_moves(sgf_file)
    print("move", move)
    now = time.time()
    info = get_position_info_evalsgf(sgf_file, move, visits)
    print("info", info)
    print("Time taken:", time.time() - now)


if __name__ == "__main__":
    _example_sgf_to_data()
    _example_analyze_ogs()
