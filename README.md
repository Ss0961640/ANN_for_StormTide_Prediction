# 人工智慧模式於近海暴潮預測之應用

本專案以 **C++ 從零實作三層前饋式類神經網路（ANN, Feedforward Neural Network）**，用於近海暴潮（Storm Tide）之潮位趨勢預測。  
核心重點在於：**不依賴現成深度學習框架**，以工程化方式完成資料正規化、矩陣運算、前向傳播推論與輸出反正規化，並將預測結果以時間序列形式輸出，作為海洋災害預警/風險評估的前端模型展示。

---

## 專案簡介（Overview）

近海暴潮常由天文潮與氣象因素共同造成，潮位短時間內可能出現顯著上升。本專案以潮汐/潮位歷史資料為基礎，建立 **三層 ANN（Input → Hidden → Output）** 模型，透過前向傳播計算輸出未來潮位值，用於觀察短期潮位變化趨勢。

本專案著重於「模型推論流程」與「工程實作細節」，包含：
- 資料正規化/反正規化（Normalization / Inverse Normalization）
- 權重矩陣與偏差向量的載入與管理（Weights / Bias）
- 前向傳播推論（Forward Propagation）
- 輸出結果寫入檔案並可進一步畫出折線圖比對趨勢

本專案聚焦於模型與預測流程實作，不涵蓋災害影響評估、淹水範圍推估與決策支援系統。

---

## 目標（Objective）

- 建立三層前饋式 ANN 架構（Input / Hidden / Output），用於潮位時間序列預測  
- 以 C++ 實作矩陣乘法與向量運算，完成前向推論流程  
- 讀取外部設定檔（網路結構、權重、偏差、資料範圍），提升模型可重現性與可移植性  
- 對輸入潮位資料進行正規化，使數值落在 [-1, 1] 以提升數值穩定性  
- 產出未來一小時潮位預測值，並可進一步用折線圖比較觀測/預測趨勢

---

## 模型架構（Model Architecture）

<img width="512" height="331" alt="螢幕擷取畫面 2026-02-26 163224" src="https://github.com/user-attachments/assets/18f1fa2d-4a73-44ed-84bf-f89914fd8897" />


三層前饋式網路（Input → Hidden → Output）：

- Input Layer（M 維）：潮位相關輸入特徵（例如當前/歷史潮位、時間窗特徵等）  
- Hidden Layer（N 個神經元）：透過非線性激活函數進行特徵轉換與映射  
- Output Layer（L 維）：輸出預測潮位（本專案常見為 1 維，即單步潮位值）  

註：網路結構（M, N, L）由 `ANNSFM_Config_S` 控制，便於調整神經元數量與輸入/輸出維度。


---

## ANN 計算流程（Forward Propagation）

本專案採用「正規化 → 隱藏層非線性映射 → 線性輸出 → 反正規化」流程。

<img width="507" height="341" alt="螢幕擷取畫面 2026-02-26 163425" src="https://github.com/user-attachments/assets/c4c4ef9c-430b-4c44-ae13-3993772fde24" />


### Input Layer（Normalization）

將原始輸入 x 映射到 [-1, 1]，避免不同量綱造成訓練/推論不穩定：

X = 2 * (x - min) / (max - min) - 1

其中 `min/max` 由 `ANNSFM_Config_I` 提供（輸入資料範圍）。

### Hidden Layer（Nonlinear Mapping）

隱藏層計算：

Hn = f( wH(n, m) * Im + bH(n) )

激活函數採用雙曲正切型式（tanh 等價形式）：

f(x) = 2 / (1 + e^(-2x)) - 1

- `wH`：Input → Hidden 權重矩陣（由 `ANNSFM_CS_HW` 讀取）  
- `bH`：Hidden Bias（由 `ANNSFM_CS_HB` 讀取）

### Output Layer（Linear Output + Inverse Normalization）

輸出層先做線性組合：

Ol = wO(l, n) * Hn + bO(l)

再進行反正規化回到原始潮位尺度：

y = (Y + 1) / 2 * (max - min) + min

其中 `min/max` 由 `ANNSFM_Config_O` 提供（輸出資料範圍）。

---

## 檔案說明（File Description）

本專案使用「設定檔 + 權重檔 + 輸入資料檔」方式，使模型推論可重現且容易替換資料/權重。

### 輸入檔案（Input Files）

| 檔案名稱 | 格式 | 說明 |
| --- | --- | --- |
| ANNSFM_Config_I | .txt | 輸入資料正規化範圍（min/max） |
| ANNSFM_Config_O | .txt | 輸出資料反正規化範圍（min/max） |
| ANNSFM_Config_S | .txt | 網路結構設定（M, N, L） |
| ANNSFM_CS_HB | .txt | Hidden Bias（隱藏層偏差） |
| ANNSFM_CS_HW | .txt | Input → Hidden 權重矩陣 |
| ANNSFM_CS_OB | .txt | Output Bias（輸出層偏差） |
| ANNSFM_CS_OW | .txt | Hidden → Output 權重矩陣 |
| ANNSFM_data_size | .txt | 資料筆數 P（輸入樣本數） |
| ANNSFM_inputs | .txt | 模型輸入資料（潮位/潮汐資料） |

### 輸出檔案（Output Files）

| 檔案名稱 | 格式 | 說明 |
| --- | --- | --- |
| ANNSFM_outputs | .txt | 模型輸出結果（預測未來一小時潮位） |

---

## 技術亮點（Technical Highlights）

- 以 C++ 從零實作三層 ANN 前向傳播推論流程（不依賴框架）  
- 完整包含資料正規化/反正規化，強化數值穩定性與可重現性  
- 以外部設定檔管理網路結構與權重/偏差，便於替換模型與資料  
- 以時間序列形式輸出預測潮位，可直接用於折線圖與誤差分析  
- 適合作為海洋災害預警系統中的「快速推論模組」或「前端展示模型」

---

## 未來延伸（Future Work）

- 多步長預測（multi-step forecasting）：由單步延伸至未來多小時潮位序列  
- 引入時間序列深度模型（LSTM / GRU / Transformer）以捕捉長期依賴  
- 多變數預測：融合風速、氣壓、降雨、浪高等特徵建立多模態模型  
- 建立即時資料串接（API/串流），提升預警系統的即時性  
- 增加評估指標（MAE/RMSE/Correlation）與觀測值對照圖，形成可交付報告

---

## 聯絡作者

GitHub: https://github.com/RaymondYang

Email: Ss0961640@gmail.com

