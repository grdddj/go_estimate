import asyncio
import json
import logging
import random
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from helpers import (
    ANALYSIS_CONFIG,
    KATAGO,
    KATAGO_MODEL_BIN,
    GameInputData,
    PositionInfo,
    get_position_info_from_json_output,
    sgf_to_data,
)

HERE = Path(__file__).parent

LOGGER_FILE = HERE / "server.log"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(LOGGER_FILE))

app = FastAPI()

katago_process = subprocess.Popen(
    [
        KATAGO,
        "analysis",
        "-model",
        KATAGO_MODEL_BIN,
        "-config",
        ANALYSIS_CONFIG,
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
)


async def send_request_to_katago(request: GameInputData) -> dict:
    request_id = "request_" + str(hash(json.dumps(request)))
    request_json = json.dumps(
        {
            "id": request_id,
            "boardXSize": 19,
            "boardYSize": 19,
            "initialStones": request["initial_stones"],
            "moves": request["moves"],
            "rules": request["rules"],
            "komi": request["komi"],
            "maxVisits": request["max_visits"],
        }
    )

    assert katago_process.stdin

    # Write the request to KataGo's stdin
    katago_process.stdin.write(request_json + "\n")
    katago_process.stdin.flush()

    # Read KataGo's response until we get the expected id for the second time
    while True:
        assert katago_process.stdout
        line = await asyncio.to_thread(katago_process.stdout.readline)
        if line:
            response = json.loads(line)
            if response.get("id") == request_id:
                return response


class AnalysisRequest(BaseModel):
    initial_stones: list
    moves: list
    rules: str = "japanese"
    komi: float = 6.5
    max_visits: int = 50


# Define the FastAPI endpoint for analysis
@app.post("/analyze")
async def analyze(request: AnalysisRequest) -> PositionInfo:
    try:
        request_id = str(random.randint(0, 100000000))
        now = time.time()
        logger.info(f"Analysis request - {request_id} - {request}")
        # Convert request object to dictionary
        request_data: GameInputData = {
            "initial_stones": request.initial_stones,
            "moves": request.moves,
            "rules": request.rules,
            "komi": request.komi,
            "max_visits": request.max_visits,
        }

        # Send the request to KataGo and get the response
        json_response = await send_request_to_katago(request_data)

        info = get_position_info_from_json_output(json_response)
        logger.info(
            f"Request finished - {request_id} - {info} - took: {time.time() - now} s"
        )
        if not info:
            raise HTTPException(status_code=500, detail="Invalid response from KataGo")
        return info

    except Exception as e:
        logger.exception(str(e))
        raise HTTPException(status_code=500, detail=str(e))


class SgfRequest(BaseModel):
    sgf: str
    visits: int


# Define the FastAPI endpoint for analysis
@app.post("/sgf")
async def sgf(request: SgfRequest) -> PositionInfo:
    # import random

    # return PositionInfo(scoreLead=30 * random.random() - 30, moveInfos=[])

    try:
        request_id = str(random.randint(0, 100000000))
        now = time.time()
        logger.info(f"SGF request - {request_id} - {request}")
        sgf = request.sgf
        visits = request.visits

        data = sgf_to_data(sgf, visits)
        json_response = await send_request_to_katago(data)

        info = get_position_info_from_json_output(json_response)
        logger.info(
            f"Request finished - {request_id} - {info} - took: {time.time() - now} s"
        )
        if not info:
            raise HTTPException(status_code=500, detail="Invalid response from KataGo")
        return info

    except Exception as e:
        logger.exception(str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Close the KataGo process when shutting down
@app.on_event("shutdown")
def shutdown_event():
    if katago_process:
        assert katago_process.stdin
        assert katago_process.stdout
        katago_process.stdin.close()
        katago_process.stdout.close()
        katago_process.terminate()
