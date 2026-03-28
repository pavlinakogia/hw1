# Homework 1: From Data to Intelligent Model
**Course:** AI Hands-on , NTUA 

**Domain:** Weather Prediction (Rain in Australia)

**Name:** Pavlina Kogia

**ID:** 09325011

## 1. Problem & Dataset Description
Το αντικείμενο της εργασίας είναι η πρόβλεψη βροχόπτωσης για την επόμενη ημέρα (`RainTomorrow`).
- **Dataset:** Χρησιμοποιήθηκε ένα υποσύνολο του "Rain in Australia" dataset με **10.000 γραμμές** και **23 στήλες**.
- **Target Variable:** `RainTomorrow` (Binary Classification: Yes/No).
- **Features:** Περιλαμβάνουν μετεωρολογικές μετρήσεις όπως θερμοκρασία, υγρασία, πίεση και ταχύτητα ανέμου.

## 2. Preprocessing Approach
Ακολουθήθηκε η στρατηγική **"Split First, Preprocess Second"** για την διασφάλιση της εγκυρότητας του μοντέλου και την αποφυγή data leakage:
- **Missing Values:** Συμπληρώθηκαν με τον **Median** (για αριθμητικά) και το **Mode** (για κατηγορικά) των δεδομένων εκπαίδευσης. Η επιλογή του median έγινε για να αποφευχθεί η επίδραση των outliers στη μέση τιμή.
- **Outliers:** Εντοπίστηκαν με τη μέθοδο IQR και εφαρμόστηκε **Winsorization (clipping)** στα όρια [Q1 - 1.5*IQR, Q3 + 1.5*IQR] ώστε να διατηρηθεί η πληροφορία χωρίς να χαλάει το scaling.
- **Encoding:** - Binary encoding (0/1) για τα `RainToday` και `RainTomorrow`.
- **One-Hot Encoding** για τα κατηγορικά χαρακτηριστικά (`Location`, `WindDir`).
- **Scaling:** Χρησιμοποιήθηκε ο **StandardScaler**. Οι παράμετροι υπολογίστηκαν στο Train set και εφαρμόστηκαν στα Val/Test sets.

## 3. Feature Engineering
Δημιουργήθηκαν 4 νέα χαρακτηριστικά:
1. **Month:** Εξαγωγή μήνα από την ημερομηνία, καθώς η βροχόπτωση ακολουθεί εποχικότητα.
2. **TempRange:** Διαφορά `MaxTemp` - `MinTemp`. Μεγάλες διαφορές συχνά υποδηλώνουν αλλαγή μετώπων.
3. **HumidityDiff:** Διαφορά υγρασίας μεταξύ 3μμ και 9πμ, που αποτελεί δείκτη συσσώρευσης υδρατμών.
4. **PressureDiff:** Διαφορά ατμοσφαιρικής πίεσης, η οποία σχετίζεται με χαμηλά βαρομετρικά.

## 4. PCA Insights
Από την ανάλυση PCA (Principal Component Analysis) προέκυψαν τα εξής:
- **Scree Plot:** Παρατηρήθηκε ότι οι πρώτες 10-15 συνιστώσες εξηγούν το μεγαλύτερο ποσοστό της διακύμανσης (~80%).
- **2D Projection:** Το scatter plot των δύο πρώτων συνιστωσών έδειξε σημαντική επικάλυψη (overlap) μεταξύ των κλάσεων (Rain/No Rain), γεγονός που επιβεβαιώνει ότι το πρόβλημα είναι σύνθετο και μη γραμμικά διαχωρίσιμο.
- **Dominant Features:** Χαρακτηριστικά όπως η υγρασία (Humidity3pm) και η πίεση (Pressure) εμφάνισαν τα μεγαλύτερα βάρη (loadings) στις πρώτες κύριες συνιστώσες.

## 5. Model Comparison
Η σύγκριση έγινε στο Test Set (980 δείγματα) που δεν χρησιμοποιήθηκε καθόλου κατά την εκπαίδευση:

| Metric | Classical (Random Forest) | Neural Network |
| :--- | :---: | :---: |
| **Accuracy** | 0.8031 | **0.8429** |
| **Precision** | 0.5699 | **0.7053** |
| **Recall** | **0.6853** | 0.5776 |
| **F1-Score** | 0.6223 | **0.6351** |
| **ROC-AUC** | 0.8556 | **0.8609** |

### Insights:
- **Hyperparameter Tuning:** Το Random Forest βελτιστοποιήθηκε μέσω `GridSearchCV` με βέλτιστες παραμέτρους: `max_depth: 10`, `min_samples_split: 5`, `n_estimators: 200`.
- **Νικητής:** Το **Neural Network** αναδείχθηκε ως το καλύτερο μοντέλο, υπερέχοντας στο F1-Score, το ROC-AUC και το Accuracy. 
- **Σχολιασμός:** Το αποτέλεσμα δεν προκαλεί έκπληξη, καθώς το νευρωνικό δίκτυο (με dropout, 2 hidden layers και early stopping) κατάφερε να γενικεύσει καλύτερα στις μη γραμμικές σχέσεις του καιρού σε ένα dataset 10.000 δειγμάτων.

## 6. Best Model Designation
Το μοντέλο που αποθηκεύτηκε ως `best_model.pkl` είναι το **Neural Network**.
**Αιτιολόγηση:** Η επιλογή βασίστηκε πρωτίστως στο **F1-Score (0.6351)** και το **ROC-AUC (0.8609)**. Σε ένα imbalanced dataset όπως αυτό της βροχής (όπου οι μέρες χωρίς βροχή είναι πλειοψηφία), το F1-Score αποτελεί το πλέον αξιόπιστο κριτήριο καθώς ισορροπεί την ακρίβεια των προβλέψεων (Precision) με την ικανότητα εντοπισμού των πραγματικών περιπτώσεων βροχής (Recall).

## 7. Installation & Execution
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/pavlinakogia/hw1]
   cd hw1
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the core pipeline (Preprocessing, Training, Evaluation):**
   ```bash
   python main.py

4. **Run the REST API (Bonus Task):**
   ```bash
   python api.py
   
Μόλις ξεκινήσει ο server, ανοίξτε τον browser σας στη διεύθυνση http://127.0.0.1:8000/docs για να δοκιμάσετε το endpoint /predict μέσω του Swagger UI.

5. **Run the interactive app Streamlit UI (Additional Task):**
   ```bash
   streamlit run app_ui.py
   
Αυτό θα ανοίξει αυτόματα μια νέα καρτέλα στον browser σας, όπου μπορείτε να πειράξετε τις τιμές του καιρού με sliders και να δείτε την πρόβλεψη του μοντέλου σε πραγματικό χρόνο.
