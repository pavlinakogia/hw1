# Homework 1: From Data to Intelligent Model
**Course:** Hands-on AI, NTUA  
**Domain:** Weather Prediction (Rain in Australia)

## 1. Problem & Dataset Description
Το αντικείμενο της εργασίας είναι η πρόβλεψη βροχόπτωσης για την επόμενη ημέρα (`RainTomorrow`).
- **Dataset:** Χρησιμοποιήθηκε ένα υποσύνολο του "Rain in Australia" dataset με **10.000 γραμμές** και **23 στήλες**.
- **Target Variable:** `RainTomorrow` (Binary Classification: Yes/No).
- **Features:** Περιλαμβάνουν μετεωρολογικές μετρήσεις όπως θερμοκρασία, υγρασία, πίεση και ταχύτητα ανέμου.

## 2. Preprocessing Approach
Ακολουθήθηκε η στρατηγική **"Split First, Preprocess Second"** για την διασφάλιση της εγκυρότητας του μοντέλου:
- **Missing Values:** Συμπληρώθηκαν με τον **Median** (για αριθμητικά) και το **Mode** (για κατηγορικά) των δεδομένων εκπαίδευσης. Αυτή η επιλογή έγινε για να αποφευχθεί η επίδραση των outliers στη μέση τιμή.
- **Outliers:** Εντοπίστηκαν με τη μέθοδο IQR και εφαρμόστηκε **Winsorization (clipping)** στα όρια [Q1 - 1.5*IQR, Q3 + 1.5*IQR] ώστε να διατηρηθεί η πληροφορία χωρίς να στρεβλώνεται το scaling.
- **Encoding:** - Binary encoding (0/1) για τα `RainToday` και `RainTomorrow`.
  - **One-Hot Encoding** για τα κατηγορικά χαρακτηριστικά (`Location`, `WindDir`).
- **Scaling:** Χρησιμοποιήθηκε ο **StandardScaler**. Οι παράμετροι υπολογίστηκαν αποκλειστικά στο Train set και εφαρμόστηκαν στα Val/Test sets για την αποφυγή data leakage.

## 3. Feature Engineering
Δημιουργήθηκαν 4 νέα χαρακτηριστικά για την ενίσχυση της προγνωστικής ισχύος:
1. **Month:** Εξαγωγή μήνα από την ημερομηνία, καθώς η βροχόπτωση ακολουθεί έντονη εποχικότητα.
2. **TempRange:** Διαφορά `MaxTemp` - `MinTemp`. Μεγάλες διαφορές συχνά υποδηλώνουν αλλαγή μετώπων.
3. **HumidityDiff:** Διαφορά υγρασίας μεταξύ 3μμ και 9πμ, που αποτελεί δείκτη συσσώρευσης υδρατμών.
4. **PressureDiff:** Διαφορά ατμοσφαιρικής πίεσης, η οποία σχετίζεται άμεσα με την έλευση χαμηλών βαρομετρικών.

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
- **Νικητής:** Το **Neural Network** αναδείχθηκε ως το καλύτερο μοντέλο, υπερέχοντας στο F1-Score και το Accuracy. 
- **Σχολιασμός:** Το αποτέλεσμα δεν προκαλεί έκπληξη, καθώς το νευρωνικό δίκτυο (με dropout και 2 hidden layers) κατάφερε να μοντελοποιήσει καλύτερα τις μη γραμμικές σχέσεις του καιρού σε ένα dataset 10.000 δειγμάτων.

## 6. Best Model Designation
Το μοντέλο που αποθηκεύτηκε ως `best_model.pkl` είναι το **Neural Network**.
**Αιτιολόγηση:** Η επιλογή βασίστηκε στο **F1-Score (0.6351)** και το **ROC-AUC (0.8609)**. Σε ένα imbalanced dataset όπως αυτό της βροχής, το F1-Score αποτελεί το πλέον αξιόπιστο κριτήριο καθώς ισορροπεί την ακρίβεια των προβλέψεων (Precision) με την ικανότητα εντοπισμού των πραγματικών περιπτώσεων βροχής (Recall).

## 7. Installation & Execution
1. Κλωνοποιήστε το repository:
   ```bash
   git clone [https://github.com/pavlinakogia/hw1]