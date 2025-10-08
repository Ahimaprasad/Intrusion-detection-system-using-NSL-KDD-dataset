import os
import timeit
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import Parallel, delayed
from termcolor import colored
import sys


warnings.filterwarnings('default')
np.random.seed(100)


DATASET_DIR = os.getcwd() 
TRAIN_FILENAME = 'KDDTrain+.txt'
TEST_FILENAME = 'KDDTest+.txt'
ATTACK_TYPES_FILENAME = 'training_attack_types.txt'


GA_POP_SIZE = 8
GA_GENERATIONS = 3
GA_MUTATION_PROB = 0.05
GA_CROSSOVER_PROB = 0.6
GA_CV_FOLDS = 3


def show_and_close_plot(delay=0.15):
    try:
        plt.show()

    except Exception:
        
        try:
            plt.close()
        except Exception:
            pass


print("Script started at", timeit.default_timer())
print("Python version:", sys.version.splitlines()[0])
print("Working directory:", os.getcwd())
print("Dataset directory set to:", DATASET_DIR)


train_file = os.path.join(DATASET_DIR, TRAIN_FILENAME)
test_file = os.path.join(DATASET_DIR, TEST_FILENAME)
attack_file = os.path.join(DATASET_DIR, ATTACK_TYPES_FILENAME)
print(f"Expecting train: {train_file}\n        test: {test_file}\n        attacks: {attack_file}")


header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type',
                'success_pred']

col_names = np.array(header_names)
nominal_idx = [1, 2, 3]
binary_idx = [6, 11, 13, 14, 20, 21]

numeric_idx = [i for i in range(41) if i not in nominal_idx and i not in binary_idx]
nominal_cols = col_names[nominal_idx].tolist()
binary_cols = col_names[binary_idx].tolist()
numeric_cols = col_names[numeric_idx].tolist()


category = defaultdict(list)
category['benign'].append('normal')
try:
    if os.path.exists(attack_file):
        print("Reading attack types file...")
        with open(attack_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    attack, cat = parts[0], parts[1]
                    category[cat].append(attack)
    else:
        print(f"Attack mapping file not found at {attack_file}. Using default mapping (normal->benign).")
    attack_mapping = {v: k for k in category for v in category[k]}
    print("Attack mappings keys sample:", list(attack_mapping.items())[:5])
except Exception as e:
    print("Error reading attack types file:", e)
    attack_mapping = {'normal': 'benign'}


try:
    print("Loading training data...")
    train_df = pd.read_csv(train_file, names=header_names)
    print("Loading test data...")
    test_df = pd.read_csv(test_file, names=header_names)
    print("Loaded. Shapes -> train:", train_df.shape, "test:", test_df.shape)
except FileNotFoundError as e:
    print("Dataset files not found. Please ensure the files exist. Error:", e)
    raise
except Exception as e:
    print("Error loading datasets:", e)
    raise


train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping.get(x, 'benign'))
test_df['attack_category'] = test_df['attack_type'].map(lambda x: attack_mapping.get(x, 'benign'))


for df in (train_df, test_df):
    if 'success_pred' in df.columns:
        df.drop(['success_pred'], axis=1, inplace=True)


allowed_cats = ['benign', 'dos']
train_df = train_df[train_df['attack_category'].isin(allowed_cats)].reset_index(drop=True)
test_df = test_df[test_df['attack_category'].isin(allowed_cats)].reset_index(drop=True)
print("After filtering to benign & dos -> train rows:", len(train_df), "test rows:", len(test_df))


print("Train attack category counts:\n", train_df['attack_category'].value_counts())


for df in (train_df, test_df):
    if 'su_attempted' in df.columns:
        df['su_attempted'] = df['su_attempted'].replace(2, 0)
    if 'num_outbound_cmds' in df.columns:
        df.drop('num_outbound_cmds', axis=1, inplace=True)
        if 'num_outbound_cmds' in numeric_cols:
            try:
                numeric_cols.remove('num_outbound_cmds')
            except ValueError:
                pass


train_Y = train_df['attack_category'].copy()
train_x_raw = train_df.drop(['attack_category', 'attack_type'], axis=1)
test_Y = test_df['attack_category'].copy()
test_x_raw = test_df.drop(['attack_category', 'attack_type'], axis=1)
print("Prepared raw X/y. train_x_raw shape:", train_x_raw.shape)



def genetic_feature_selection(X_train, y_train, pop_size=GA_POP_SIZE, generations=GA_GENERATIONS):
    X_train = X_train.copy()
    num_features = X_train.shape[1]
    if num_features == 0:
        return []

    
    population = [np.random.randint(0, 2, num_features).tolist() for _ in range(pop_size)]

    skf = StratifiedKFold(n_splits=GA_CV_FOLDS, shuffle=True, random_state=42)

    def fitness(individual):
        
        selected_idx = [i for i in range(num_features) if individual[i] == 1]
        if not selected_idx:
            return 0.0
        selected_cols = [X_train.columns[i] for i in selected_idx]
        X_sel = X_train[selected_cols].copy()

       
        nom_sel = [c for c in selected_cols if c in nominal_cols]
        if nom_sel:
            X_sel = pd.get_dummies(X_sel, columns=nom_sel, drop_first=True)

       
        num_sel = [c for c in X_sel.columns if c in numeric_cols]
        if num_sel:
            try:
                scaler = StandardScaler().fit(X_sel[num_sel])
                X_sel[num_sel] = scaler.transform(X_sel[num_sel])
            except Exception:
                pass

        clf = LogisticRegression(random_state=42, max_iter=200)
        try:
            scores = cross_val_score(clf, X_sel, y_train, cv=skf, scoring='accuracy', n_jobs=1)
            return scores.mean()
        except Exception:
            return 0.0

  
    for gen in range(generations):
        print(f"GA generation {gen+1}/{generations} - evaluating population of size {len(population)}")
        fitnesses = Parallel(n_jobs=-1)(delayed(fitness)(ind) for ind in population)

       
        def select(pop, fits, k=3):
            selected = []
            for _ in range(len(pop)):
                idxs = np.random.randint(0, len(pop), k)
                best = idxs[np.argmax([fits[i] for i in idxs])]
                selected.append(pop[best])
            return selected

        parents = select(population, fitnesses)

        # crossover
        offspring = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i+1] if (i+1) < len(parents) else parents[0]
            if np.random.rand() < GA_CROSSOVER_PROB:
                cx = np.random.randint(1, num_features-1)
                c1 = p1[:cx] + p2[cx:]
                c2 = p2[:cx] + p1[cx:]
            else:
                c1, c2 = p1[:], p2[:]
            offspring.extend([c1, c2])

        # mutation
        for ind in offspring:
            for i in range(num_features):
                if np.random.rand() < GA_MUTATION_PROB:
                    ind[i] = 1-ind[i]

        population = offspring[:pop_size]

    # evaluate final population
    fitnesses = Parallel(n_jobs=-1)(delayed(fitness)(ind) for ind in population)
    best_idx = int(np.argmax(fitnesses))
    best_individual = population[best_idx]
    selected_features_idx = [i for i in range(num_features) if best_individual[i] == 1]
    return selected_features_idx


print("Performing Genetic Algorithm for feature selection...")
try:
    selected_idx = genetic_feature_selection(train_x_raw, train_Y)
    if not selected_idx:
        print("GA selected no features. Using all features as fallback.")
        selected_columns = train_x_raw.columns.tolist()
    else:
        selected_columns = [train_x_raw.columns[i] for i in selected_idx]
    print("Selected features (sample 20):", selected_columns[:20])
except Exception as e:
    print("Error during GA. Falling back to all features. Error:", e)
    selected_columns = train_x_raw.columns.tolist()


train_x_raw = train_x_raw[selected_columns].copy()
test_x_raw = test_x_raw[selected_columns].copy()


nominal_cols = [c for c in nominal_cols if c in selected_columns]
binary_cols = [c for c in binary_cols if c in selected_columns]
numeric_cols = [c for c in numeric_cols if c in selected_columns]

combined_df_raw = pd.concat([train_x_raw, test_x_raw], axis=0)
if nominal_cols:
    combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)
else:
    combined_df = combined_df_raw.copy()

train_x = combined_df.iloc[:len(train_x_raw)].reset_index(drop=True)
test_x = combined_df.iloc[len(train_x_raw):].reset_index(drop=True)


numeric_cols_present = [c for c in numeric_cols if c in train_x.columns]
if numeric_cols_present:
    scaler = StandardScaler().fit(train_x[numeric_cols_present])
    train_x[numeric_cols_present] = scaler.transform(train_x[numeric_cols_present])
    test_x[numeric_cols_present] = scaler.transform(test_x[numeric_cols_present])

print("Preprocessing complete. train_x shape:", train_x.shape, "test_x shape:", test_x.shape)


train_Y_bin = train_Y.map({'benign': 0, 'dos': 1})
test_Y_bin = test_Y.map({'benign': 0, 'dos': 1})


results = []



def evaluate_and_record(name, true_y, pred_y, model_obj=None, training_time=None, y_is_bin=False):
    try:
        if y_is_bin and isinstance(true_y.iloc[0], (np.integer, int)):
            # convert pred numeric to string labels for consistent reporting
            pred_labels = np.where(np.array(pred_y) == 0, 'benign', 'dos')
            true_labels = np.where(np.array(true_y) == 0, 'benign', 'dos')
        else:
            pred_labels = pred_y
            true_labels = true_y

        c_matrix = confusion_matrix(true_labels, pred_labels)
        error = zero_one_loss(true_labels, pred_labels)
        score = accuracy_score(true_labels, pred_labels)
        print(colored(f"------{name} Results-------", 'green'))
        print('Confusion Matrix:\n', c_matrix)
        print(f"Error: {error*100:.4f}%")
        print(f"Accuracy: {score*100:.4f}%")
        print(classification_report(true_labels, pred_labels))
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, xticks_rotation=45)
            plt.title(f"Confusion Matrix for {name}")
            show_and_close_plot()
        except Exception:
            pass
        results.append({'Model': name, 'Accuracy': score * 100, 'Error': error * 100, 'Training Time': training_time})
    except Exception as e:
        print(f"Error evaluating {name}: {e}")



def decision_tree_clf():
    name = 'Decision Tree'
    clf = DecisionTreeClassifier(random_state=17)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = clf.predict(test_x)
    evaluate_and_record(name, test_Y, pred, clf, t)


def random_forest_clf():
    name = 'Random Forest'
    clf = RandomForestClassifier(criterion='entropy', max_depth=30, n_estimators=48, random_state=0, n_jobs=-1)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = clf.predict(test_x)
    evaluate_and_record(name, test_Y, pred, clf, t)


def knn_clf():
    name = 'KNN'
    clf = KNeighborsClassifier(n_neighbors=7, n_jobs=None)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = clf.predict(test_x)
    evaluate_and_record(name, test_Y, pred, clf, t)


def svm_clf():
    name = 'SVM'
    clf = SVC(kernel='linear', degree=1, C=3, probability=True)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = clf.predict(test_x)
    evaluate_and_record(name, test_Y, pred, clf, t)


def logistic_reg_clf():
    name = 'Logistic Regression'
    clf = LogisticRegression(C=1e5, random_state=0, max_iter=200)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = clf.predict(test_x)
    evaluate_and_record(name, test_Y, pred, clf, t)


def sgd_clf():
    name = 'SGD'
    clf = SGDClassifier(loss='hinge', penalty='l1', max_iter=200, alpha=0.001, random_state=0)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = clf.predict(test_x)
    evaluate_and_record(name, test_Y, pred, clf, t)


def adaboost_clf():
    name = 'AdaBoost'
    clf = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=42)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = clf.predict(test_x)
    evaluate_and_record(name, test_Y, pred, clf, t)


def mlp_clf():
    name = 'MLP'
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(256, 32), max_iter=500, random_state=1)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = clf.predict(test_x)
    evaluate_and_record(name, test_Y, pred, clf, t)


def xgboost_clf():
    name = 'XGBoost'
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=0,
                            learning_rate=0.1, max_depth=3, n_estimators=200)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y_bin)
    t = timeit.default_timer() - start
    pred_num = clf.predict(test_x)
    pred_labels = np.where(np.array(pred_num) == 0, 'benign', 'dos')
    evaluate_and_record(name, test_Y, pred_labels, clf, t, y_is_bin=True)


def votingClassifier():
    name = 'Voting Classifier'
    rf = RandomForestClassifier(criterion='entropy', max_depth=30, n_estimators=48, random_state=0)
    lr = LogisticRegression(max_iter=200)
    knn = KNeighborsClassifier(n_neighbors=7)
    model = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('knn', knn)], voting='soft', weights=[2,1,1], n_jobs=-1)
    start = timeit.default_timer()
    model.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = model.predict(test_x)
    evaluate_and_record(name, test_Y, pred, model, t)


def lightgbm_clf():
    name = 'LightGBM'
    clf = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=1000, n_jobs=-1)
    start = timeit.default_timer()
    clf.fit(train_x, train_Y_bin)
    t = timeit.default_timer() - start
    pred_num = clf.predict(test_x)
    pred_labels = np.where(np.array(pred_num) == 0, 'benign', 'dos')
    evaluate_and_record(name, test_Y, pred_labels, clf, t, y_is_bin=True)


def catboost_clf():
    name = 'CatBoost'
    # CatBoost expects categorical feature indices relative to the dataframe passed to fit
    try:
        cat_features_idx = []
        for c in nominal_cols:
            if c in train_x.columns:
                cat_features_idx.append(train_x.columns.get_loc(c))
    except Exception:
        cat_features_idx = []

    clf = cb.CatBoostClassifier(iterations=200, cat_features=cat_features_idx, learning_rate=0.1,
                                l2_leaf_reg=3, max_depth=6, verbose=0, random_state=42)
    start = timeit.default_timer()
    # CatBoost can accept string labels directly
    clf.fit(train_x, train_Y)
    t = timeit.default_timer() - start
    pred = clf.predict(test_x)
    evaluate_and_record(name, test_Y, pred, clf, t)

if __name__ == '__main__':
    try:
        # Run a selection of models (you can comment/uncomment to speed up testing)
        decision_tree_clf()
        random_forest_clf()
        knn_clf()
        svm_clf()
        logistic_reg_clf()
        sgd_clf()
        adaboost_clf()
        mlp_clf()
        xgboost_clf()
        votingClassifier()
        lightgbm_clf()
        catboost_clf()

        if results:
            results_df = pd.DataFrame(results)
            print('\nConsolidated Model Performance:')
            print(results_df[['Model', 'Accuracy', 'Error', 'Training Time']].to_string(index=False))

            # Simple bar plot (non-blocking)
            try:
                plt.figure(figsize=(12, 6))
                plt.bar(results_df['Model'], results_df['Accuracy'])
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Model')
                plt.ylabel('Accuracy (%)')
                plt.title('Model Accuracy Comparison')
                plt.tight_layout()
                show_and_close_plot()
            except Exception:
                pass

    except Exception as e:
        print('Error during execution:', e)
        raise
