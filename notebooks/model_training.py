import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

## 데이터셋 로드

data_path = "../data/"

roadsurface_data = os.path.join(data_path, "road.csv" )

pre_df = pd.read_csv(roadsurface_data)

print(pre_df.head())

##----------------------------------
##            데이터 전처리
##----------------------------------

# 제거할 값 리스트
exclude_values = ["ERR", "MAX", "UNK"]

# 두 컬럼 모두 확인해서 제거
df = pre_df[
    ~pre_df["avg_road_ascat_symbl"].isin(exclude_values) & 
    ~pre_df["trinspct_road_ascat_symbl"].isin(exclude_values)
]

# "atmp_tmpr","road_tmpr","rltv_hmdt" 숫자가 아니면 NaN으로 변환
for col in ["atmp_tmpr", "road_tmpr", "rltv_hmdt"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# NaN이 있는 행 제거
df_clean = df.dropna(subset=["atmp_tmpr", "road_tmpr", "rltv_hmdt"])

##--------------------------
##     시간 데이터 추가
##--------------------------

# 1. 숫자형 피처 변환 및 NaN 제거

for col in ["atmp_tmpr", "road_tmpr", "rltv_hmdt"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 타겟 컬럼도 NaN 제거
df = df.dropna(subset=["atmp_tmpr", "road_tmpr", "rltv_hmdt", "avg_road_ascat_symbl", "ocrr_dt"])


# 2. 시간(datetime) 처리

df["ocrr_dt"] = pd.to_datetime(df["ocrr_dt"], errors="coerce")

# NaT 제거
df = df.dropna(subset=["ocrr_dt"])

# 시간대(time slot) 컬럼 생성
def time_slot(hour):
    if 0 <= hour < 6:
        return "midnight"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"

df["time_slot"] = df["ocrr_dt"].dt.hour.apply(time_slot)

# 원-핫 인코딩
df = pd.get_dummies(df, columns=["time_slot"])

# 3. 피처와 타겟 정의
feature_cols = ["atmp_tmpr", "road_tmpr", "rltv_hmdt"] + [c for c in df.columns if c.startswith("time_slot_")]
X = df[feature_cols]
y = df["avg_road_ascat_symbl"]

# NaN 또는 무한대 제거
import numpy as np
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

print("전처리 완료, 피처 수:", X.shape[1], "샘플 수:", X.shape[0])


##---------------------------
##    오버 샘플링 및 데이터 학습
##---------------------------

from imblearn.over_sampling import SMOTE

# ------------------------
# 1. 학습/테스트 데이터 분리
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------
# 2. SMOTE 오버샘플링 (train 데이터에만 적용)
# ------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:\n", y_train.value_counts())
print("After SMOTE:\n", y_train_res.value_counts())

# ------------------------
# 3. RandomForest 학습
# ------------------------
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train_res, y_train_res)

# ------------------------
# 4. 예측 및 평가
# ------------------------
y_pred = model.predict(X_test)

print("📊 분류 리포트:\n", classification_report(y_test, y_pred))
print("📊 혼동 행렬:\n", confusion_matrix(y_test, y_pred))

# ------------------------
# 5. 예측 확률 저장
# ------------------------
proba = model.predict_proba(X_test)
df_results = X_test.copy() 
df_results["predicted_state"] = y_pred
df_results["true_state"] = y_test.values