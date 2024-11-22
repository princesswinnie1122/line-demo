# UniHelp
> If you need help, go for UniHelp.

## 動機 (Motivation)

隨著國際學生在校園中數量的增加，他們在適應新環境、理解課程安排以及與當地學生和老師的溝通上經常面臨困難。語言障礙和資訊不對稱成為他們融入校園生活的主要挑戰。因此，我們希望開發一個能夠自動化處理這些問題的工具，幫助國際學生在學業和生活中獲得支持，並讓他們能更快融入校園環境。這個專案就是為了解決這些需求而生——UniHelp，一個整合語音、圖像和課程推薦功能的智能聊天機器人。

## 目標對象 (Target Audience, TA)

- 國際學生：這些學生在語言、課程選擇、學校資源了解等方面面臨困難。
- 交換學生：短期在外學習的學生希望能快速適應並了解學校相關資源。
- 跨國公司員工：需要跨國溝通與適應新環境的外派員工。

## 目標問題 (Target Problem)

- 語言障礙：國際學生面對的最大挑戰是與當地語言和文化的障礙，導致他們無法有效溝通和理解課程內容。
- 課程推薦和選擇困難：國際學生對於學校內的課程安排和評價不熟悉，難以做出最佳的選課決定。
- 資訊過載：國際學生經常會被大量的資料和信息淹沒，缺乏一個簡潔、可理解的方式來獲取所需的資訊。

## 內容簡介 (Project Overview)

- 語音轉文字：通過語音識別技術，國際學生可以用自己的母語進行詢問，並獲得即時的文字回應，打破語言障礙。
- 圖像資訊整理：學生可以上傳拍攝的課程筆記或公告圖像，UniHelp 將自動整理並翻譯成所需語言，方便學生理解。
- 課程推薦：根據學生的興趣、課程評價以及過往的選課紀錄，UniHelp 會給出個性化的課程推薦。
- 多語言切換：UniHelp 可以根據不同國家和地區的用戶自動切換語言，支援中文、英文、法文、日文等多種語言，確保使用體驗的無縫對接。

## 技術架構 (Technical Architecture)

- 大規模語言模型 API：主要使用 LLM API（OpenAI）來進行自然語言處理。通過調用這些 API，我們實現了語音轉文字、文字生成、多語言翻譯等核心功能。
- 語音處理技術：使用語音識別技術將用戶的語音訊息轉換為文字，進而進行語義分析，提供相應的回答。
- 圖像處理技術：利用圖像識別與文字提取（如 OCR 技術），將上傳的圖像資料轉換為可讀文字，並整合翻譯功能。
- 課程推薦算法：根據學生輸入的興趣、課程評價、以往的選課數據，設計推薦模型，提供精準的課程建議。
- 多語言支援：通過 LLM 的 API 進行即時翻譯，根據不同地區的學生自動切換對應的語言。

## 預期成效 (Expected Outcomes)

- 提升國際學生的學習效率：透過即時翻譯和課程推薦功能，國際學生能夠更容易獲得所需資訊，減少語言障礙帶來的困難。
- 減少信息混亂：國際學生能夠輕鬆整理課程資訊和活動公告，避免資訊過載的情況，專注於學業。
- 提高跨文化交流能力：學生能夠使用 UniHelp 進行多語言對話，加強與當地學生和教師的互動，增進學習體驗。
- 未來展望 (Future Prospects)

## 未來展望

- 更多語言支援：我們將增加更多語言支援，讓來自世界各地的學生都能夠無縫使用。
- 社群互動功能：結合社交功能，讓學生能透過 UniHelp 加入學校的社群，與其他學生交流。
- 個性化學習建議：根據學生的學習進度與表現，提供個性化的學習建議，協助他們更好地規劃學習計畫。
- 學校系統整合：與學校的教務系統進行深度整合，提供更精確的課程資料及校園活動訊息。
