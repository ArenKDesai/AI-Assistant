# AI-Assistant
Zach Gunderson and Aren Desai attempt to make AGI.

## File Structure
Use gpt to help format this

```
ai_assistant/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── input_modules/
│   │   ├── telegram_bot.py
│   │   ├── email_integration.py
│   │   └── news_scraper.py
│   ├── processing_modules/
│   │   ├── llm_processing.py
│   │   └── task_scheduler.py
│   ├── output_modules/
│   │   ├── alert_system.py
│   │   └── message_output.py
│   ├── storage/
│   │   ├── database_manager.py
│   │   └── local_storage.py
│   ├── integration_layers/
│   │   ├── api_gateway.py
│   │   └── authentication_module.py
│   ├── utils/
│   │   └── helpers.py
│   └── app.py
├── tests/
│   └── test_bot.py
├── requirements.txt
└── README.md
```

## Classes
```
Agent (Generic)
    └── Schedules (Queue of Tasks)
    └── Read with purpose (Prompt)
    └── Store Messages (SQL)
    └── Send Follow Ups

Team-Leader
    └── Orangizes Agents Tasks
    └── Manage Resources (schedules)
    └── 

Agent-Team
    └── Team-Leader schedules tasks for the Agents
        The Team-Leader takes in text and puts them in queues for the Agents
        └── Agent-Message-Attendent
            └── Tracks your messages to family and friends on text, and sends reminders (birthdays, follow ups, etc)
        └── Agent-Email-Attendent
            └── Tracks your emails and sends reminders
        └── Agent-News-Attendent
            └── Tracks your favorite news subjects and sends you daily updates
        └── Agent-Therapist
            └── Tracks your journal and moods
        └── Agent-Idea-Tracker
            └── Tracks ideas you have and stores them in subjects (lots of subjects)
                ~example 1~ Idea: Beer Squid (Beer bong gamified with embedded tech), Subject: Project Ideas 
                ~example 2~ Idea: LED controller for Guitar Amps for Aren, Subject:  Friend Gift Ideas
        └── Agent-School
            └── Tracks Canvas Assignments and gives reminders on uncompleted stuff and upcoming schedules
```
## CI/CL
I added in Continuous Integration (CI) and Continuous Linting (CL) to keep our code less error prone during this project. It's nice to do this since it's a group project.

please install: `pip install black flake8 codespell`

- **Black**: Ensures consistent formatting across all Python files.
- **Flake8**: Checks for Python code style and quality issues.
- **Codespell**: Identifies common spelling mistakes in the code.

### Running Linters and Formatters Locally

To avoid CI failures, it's recommended to run the following checks manually before pushing code:

1. **Install Required Tools**

   You can install the tools used for code quality checks via `pip`:

   ```bash
   pip install black flake8 codespell

## Git Best Practices
A little spot to put our pet peeves.

Zach:
- Issue: Please put every file into it's own commit
    Reason: This makes the history look nice and makes checking things out easy when we mess up

Aren: