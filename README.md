# Islamophobia Content Analytic System

An AI-powered web analytics system designed to detect Islamophobic content and evaluate its severity. The system utilizes a custom **Hierarchical BiLSTM (Bidirectional Long Short-Term Memory) Neural Network** with CBOW Word Embeddings to perform multi-task classification on textual data.

---

## 🚀 How to Run the System

### Step 1: Start the AI Model Server
We have provided a automated launcher that will search for Python, set up the environment, verify dependencies, and start the Flask API.

1. Locate the file **`Server_Start.bat`** in the project root directory.
2. Double-click **`Server_Start.bat`** to run it.
   * *The script will automatically detect local Python installations (such as Python 3.11) even if they are not added to your system's environment `PATH` variable.*
   * *It will check and install any missing required packages (`flask`, `flask-cors`, `torch`, `pandas`, `numpy`, `keras`).*
   * *It sets the environment backend to `KERAS_BACKEND=torch` for loading the model tokenizer.*
3. Once running, you will see a message: `* Running on http://127.0.0.1:5001`. Do not close this terminal window while using the application.

### Step 2: Open the Web Application
1. Double-click or open **`StartHere.html`** (or `index.html`) in any modern web browser.
2. Navigate through the pages to analyze text input directly or upload datasets for batch CSV processing.

---

## 📂 Project Structure

* **`TextApp.py`**: The main Flask API server. It exposes endpoints for single-text evaluation and communicates with the frontend.
* **`BatchApp.py`**: Handles processing logic for batch uploaded datasets (CSV files).
* **`Server_Start.bat`**: The automated setup & startup script for Windows.
* **`best_model_complete.pth`**: PyTorch checkpoint containing the trained model weights and tokenizer dictionary.
* **`requirements.txt`**: List of required Python packages.
* **Frontend Web Pages**:
  * `StartHere.html` / `index.html` - Welcome & landing portal.
  * `UserHomePage.html` / `AdminHomePage.html` - Role dashboards.
  * `singletextanalysisprocess.html` - Interactive single text analysis interface.
  * `uploaddatasetprocess.html` - Batch CSV upload and analysis results display.
  * `CombinedInfomationPage.html` / platform-specific pages (`RedditInformationPage.html`, `XInformationPage.html`, etc.) - Visualized model performances, graphs, confusion matrices, and project info.

---

## 📊 Model Information

The classification architecture is a **Hierarchical BiLSTM Model**:
* **Vocabulary Size**: 26,477 words
* **Embedding Dimension**: 200 (CBOW)
* **Hidden LSTM Dimension**: 256
* **Target Classification Head**: Predicts targets (**Islam**, **Muslim**, **Other**)
* **Severity Assessment Head**: Predicts severity (**Neutral**, **Nonviolent_Hate**, **Promoting Violent**)
