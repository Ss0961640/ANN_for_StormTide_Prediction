# ANN for Storm Tide Prediction (C++ Implementation)
以三層前饋式類神經網路（ANN），進行近海暴潮（Storm Tide）之潮位趨勢預測。



## 專案簡介 (Overview)
本專案以 **C++ 自行實作 ANN（Artificial Neural Network）模型**，  
透過潮汐歷史資料建立三層前饋式神經網路（Input → Hidden → Output），  
進行近海暴潮（Storm Tide）之潮位變化趨勢預測，作為海洋災害風險評估之前端模型展示。

本研究重點在於：
- AI 模型於實際海洋資料之應用
- ANN 結構設計與矩陣運算實作
- 模型輸出結果視覺化（潮位折線圖）

> 本專案聚焦於模型與預測流程，不探討災害影響評估與風險決策層面。

---

## 研究目標 (Objective)
- 建立三層前饋式 ANN 架構（Input / Hidden / Output）
- 基於權重矩陣、偏差與前向傳播計算
- 將潮汐資料正規化後輸入模型進行預測
- 視覺化預測結果觀察暴潮趨勢

---

## 模型架構 (Model Architecture)

<p align="center">
  <img src="https://github.com/user-attachments/assets/14a3bcac-8600-4b48-bb09-d643713838ea" width="600">
</p>

- Input Layer：潮汐相關特徵  
- Hidden Layer：非線性映射  
- Output Layer：預測潮位值  

---

## ANN 計算流程 (Forward Propagation)

<p align="center">
  <img src="https://github.com/user-attachments/assets/2c4a7d5d-44b4-47b9-8034-d6842402167c" width="600">
</p>

### Input Layer
1. Read data
2. Normalization: x → [-1, 1]
   X = 2*(x - min)/(max - min) - 1

### Hidden Layer
Hn = f( wH(n,m) * Im + bH(n) )
f(x) = 2/(1 + e^(-2x)) - 1

### Output Layer
Ol = wO(l,n) * Hn + bO(l)
Inverse normalization:
y = (Y + 1)/2 * (max - min) + min


---
### 檔案說明 (File Description)
輸入檔案 (Input Files)
| 檔案名稱               | 格式         | 說明                                       |
| ------------------ | ---------- | ---------------------------------------- |
| `ANNSFM_Config_I`  | 文字檔 (.txt) | 資料正規化所需之**輸入值範圍**                        |
| `ANNSFM_Config_O`  | 文字檔 (.txt) | 資料正規化所需之**輸出值範圍**                        |
| `ANNSFM_Config_S`  | 文字檔 (.txt) | 三層網路結構與神經元個數設定（M, N, L）                  |
| `ANNSFM_CS_HB`     | 文字檔 (.txt) | 隱藏層神經元之偏權值（Hidden Bias）                  |
| `ANNSFM_CS_HW`     | 文字檔 (.txt) | 輸入層 → 隱藏層 之權重連結（Input → Hidden Weights）  |
| `ANNSFM_CS_OB`     | 文字檔 (.txt) | 輸出層神經元之偏權值（Output Bias）                  |
| `ANNSFM_CS_OW`     | 文字檔 (.txt) | 隱藏層 → 輸出層 之權重連結（Hidden → Output Weights） |
| `ANNSFM_data_size` | 文字檔 (.txt) | 資料筆數 P                                   |
| `ANNSFM_inputs`    | 文字檔 (.txt) | 模型輸入資料（中央氣象署 / 氣象局資料）                    |

輸出檔案 (Output Files)
| 檔案名稱             | 格式         | 說明                            |
| ---------------- | ---------- | ----------------------------- |
| `ANNSFM_outputs` | 文字檔 (.txt) | 模型輸出結果（讀入氣象資料後，**預測未來一小時數值**） |


