

## Technology Stack

* **Python 3.8+**
* **PyTorch**
* **Ultralytics YOLOv8**
* **OpenCV**
* **Kaggle Notebooks** (for training)

---

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.8 or newer
* Git

1.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1.  **Place the Model:**
    Place the downloaded `best.pt` file in the root directory of the project.

2.  **Run the Inference Script:**
    The `inference.py` script is configured to run on your webcam by default.
    ```bash
    python inference.py
    ```
    * To use a video file instead, open `inference.py` and modify the script to point to your video file.

---

## Model Performance

The model was trained for 10 epochs on the "PPE Detection v3" dataset from Roboflow.

| Class               | Precision | Recall | mAP50 | mAP50-95 |
| :------------------ | :-------- | :----- | :---- | :------- |
| **Overall** | 0.72      | 0.715  | 0.735 | 0.456    |
| **`head_helmet`** | 0.784     | 0.824  | 0.866 | 0.584    |
| **`vest`** | 0.841     | 0.897  | 0.935 | 0.705    |

