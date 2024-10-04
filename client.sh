curl -X POST "http://127.0.0.1:8590/analyze" -H "Content-Type: application/json" -d '{
  "initial_stones": [["B", "Q9"], ["B", "C4"]],
  "moves": [["W", "P5"], ["B", "P6"]],
  "rules": "japanese",
  "komi": 6.5,
  "max_visits": 50
}'
