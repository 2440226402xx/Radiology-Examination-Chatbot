# DeepSeek REC Advanced

A modular radiology chatbot system powered by fine-tuned DeepSeek R1.

## Project Structure Overview

```
deepseek_rec_advanced/
├── app/
│   ├── api.py                      # Flask API entry point
│   ├── routes/
│   │   └── ask.py                  # API route handler
│   └── services/
│       ├── chat_service.py         # LLM chat logic
│       └── prompt_service.py       # Prompt selection per scenario
│
├── app/models/
│   └── model_loader.py             # Load fine-tuned DeepSeek R1
│
├── app/config/
│   └── config.py                   # Path and environment configs
│
├── core/
│   ├── prompts/
│   │   ├── AT_prompt.txt           # Appointment Triage prompt
│   │   ├── PP_prompt.txt           # Pre-exam Preparation prompt
│   │   └── RCS_prompt.txt          # Radiology Clinic Services prompt
│   ├── utils/
│   │   ├── logger.py               # Logging utility
│   │   ├── scenario_classifier.py  # Classify input into scenarios
│   │   └── response_formatter.py   # Postprocess outputs
│   └── evaluation/
│       └── evaluator.py            # Placeholder for evaluation logic
│
├── main.py                         # Entrypoint to launch the app
├── requirements.txt                # Python package dependencies
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

This layout is optimized for flexibility and future evaluation integration.
