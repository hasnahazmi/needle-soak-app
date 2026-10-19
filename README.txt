Needle Soak & Alignment Recipe Generator
========================================

1) Install Python 3.10+ (tick "Add Python to PATH" on Windows).

2) Open a terminal in this folder and run:

     python -m venv venv
     venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
     pip install -r requirements.txt

3) Start the app:

     python main.py

Quick test: choose "Simulated (for testing)" and press Start, or choose
"File Upload", press Start and pick sample_profile.csv. Then press
"End Run & Generate Recipe".

Temperature
-----------
"Run Info" has a Temperature dropdown (-40 to 125, 5-degree steps, default 25)
with a fixed "°C" unit beside it. The temperature is shown in the AI recipe,
sent to the AI together with your readings, and saved in the CSV export.
Run each temperature as its own soak.

The recipe
----------
After "End Run & Generate Recipe" the AI Recipe Suggestion box shows the
soak time, alignment checks, confidence, interpretation, production
recommendation and risk, with the probe card, lot/wafer, temperature and time
at the top. The recipe comes from the AI only. A one-line CROSS-CHECK compares
it with the app's own curve fit and warns if they disagree - verify before
using a recipe that shows a WARNING. Without a working AI there is no recipe
(the chart and fitted curve are still shown).

AI setup (left panel: "AI Recipe Settings")
-------------------------------------------
Option A - Microsoft Copilot (copy/paste)   <- default, no key, no IT setup
  1. Provider: "Microsoft Copilot (copy/paste)".
  2. Collect data, press "End Run & Generate Recipe".
  3. Click "Copy prompt for Copilot", paste it into Copilot chat, send.
  4. Copy Copilot's reply, paste it into the reply box, click "Apply reply".
  The app checks the reply and cross-checks it against the curve fit.
  If Copilot answers with extra text, that is fine - only the JSON is used.
  If it refuses or rambles, ask: "Reply again with only the JSON object in
  a code block."

Option B - Claude (cloud API)
  1. Get an API key from https://console.anthropic.com
  2. Paste it in the "API key" box (not saved to disk), or set the
     ANTHROPIC_API_KEY environment variable.
  3. Model: blank = claude-haiku-5-5, or claude-sonnet-5-5 for higher quality.
  NOTE: readings are sent over the internet. Check your company data policy.
  Many company networks block api.anthropic.com.

Option C - Ollama (local, data stays on the PC)
  1. Install Ollama from https://ollama.com/download
  2. Run once:  ollama pull llama3.2
  3. Pick "Ollama (local)" as the provider.

If the AI is unavailable the app still shows the curve-fit recipe.

File upload columns
-------------------
Time column : "Interval (Min)" or "Step" (else first column)
Height diff : "Z Height Difference", "Z Diff" or "Z Height" (else second column)
Supported   : .xlsx, .xls, .csv, .txt

Files
-----
main.py          entry point
ui_main.py       window, chart, buttons, AI settings
analysis.py      curve fit, soak time, alignment schedule
ai_recipe.py     AI recipe request, validation, cross-check
data_sources.py  manual / simulated / file / GPIB (stub) sources
