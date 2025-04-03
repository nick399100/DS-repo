import os
import json
import time
import pandas as pd
import sys
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

# 載入 .env 中的 GEMINI_API_KEY
load_dotenv()
#HW2 定義評分項目
# 定義客服專員評分項目
ITEMS = [
    "溝通技巧（語速與音量適當性）",
    "溝通技巧（語言表達流暢性）",
    "溝通技巧（親和力與禮貌性）",
    "問題解決（資訊確認與收集能力）",
    "問題解決（快速應對與反應力）",
    "問題解決（解決方案的適當性）",
    "專業知識（業務熟悉度）",
    "專業知識（流程規範遵循度）",
    "主動服務（需求確認與提醒）",
    "耐心與情緒管理（冷靜處理衝突）",
    "貴賓體驗提升（額外建議與推薦）"
]

def parse_response(response_text):
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        result = json.loads(cleaned)
        for item in ITEMS:
            if item not in result:
                result[item] = ""
        return result
    except Exception as e:
        print(f"解析 JSON 失敗：{e}")
        return {item: "" for item in ITEMS}

def select_dialogue_column(chunk: pd.DataFrame) -> str:
    """
    根據 CSV 欄位內容自動選取存放逐字稿的欄位。
    優先檢查常見欄位名稱："text", "utterance", "content", "dialogue"
    若都不存在，則回傳第一個欄位。
    """
    preferred = ["text", "utterance", "content", "dialogue", "Dialogue"]
    for col in preferred:
        if col in chunk.columns:
            return col
    print("CSV 欄位：", list(chunk.columns))
    return chunk.columns[0]
#HW2 評分標準的prompt的prompt
def process_batch_dialogue(client, dialogues, delimiter="-----"):
    prompt = (
        "你是一位客服對話質量分析專家，請根據以下標準評估客服專員的服務質量：\n"
        + "".join([f"{i+1}. {item}\n" for i, item in enumerate(ITEMS)]) +
        "\n請依據每個項目的具體表現，給出 1 到 5 分的評分標記：\n"
        "- 1：極差，完全不符合標準\n"
        "- 2：差，僅達到基本要求\n"
        "- 3：中等，符合基本要求但無突出\n"
        "- 4：良好，表現較佳\n"
        "- 5：優秀，遠超標準\n"
        "\n請對每筆逐字稿產生 JSON 格式回覆，並在各筆結果間用下列分隔線隔開：\n"
        f"{delimiter}\n"
        "例如：\n"
        "```json\n"
        "{\n  \"溝通技巧（語速與音量適當性）\": \"4\",\n  \"溝通技巧（語言表達流暢性）\": \"3\",\n  ...\n}\n"
        f"{delimiter}\n"
        "{{...}}\n```"
    )

    batch_text = f"\n{delimiter}\n".join(dialogues)
    content = prompt + "\n\n" + batch_text
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content
        )
        print("批次 API 回傳內容：", response.text)
        parts = response.text.split(delimiter)
        results = [parse_response(part) for part in parts]

        # 確保回傳結果數量正確
        if len(results) > len(dialogues):
            results = results[:len(dialogues)]
        elif len(results) < len(dialogues):
            results.extend([{item: "" for item in ITEMS}] * (len(dialogues) - len(results)))

        print("處理結果：")
        for result in results:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return results
    except ServerError as e:
        print(f"API 呼叫失敗：{e}")
        return [{item: "" for item in ITEMS} for _ in dialogues]

def main():
    if len(sys.argv) < 2:
        print("Usage: python customer_analysis.py <path_to_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = "customer_analysis.csv"
    if os.path.exists(output_csv):
        os.remove(output_csv)

    df = pd.read_csv(input_csv)
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("請設定環境變數 GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)

    dialogue_col = select_dialogue_column(df)
    print(f"使用欄位作為逐字稿：{dialogue_col}")

    batch_size = 10
    total = len(df)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = df.iloc[start_idx:end_idx]
        dialogues = batch[dialogue_col].tolist()
        dialogues = [str(d).strip() for d in dialogues]
        batch_results = process_batch_dialogue(client, dialogues)
        batch_df = batch.copy()
        for item in ITEMS:
            batch_df[item] = [res.get(item, "") for res in batch_results]
        if start_idx == 0:
            batch_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        else:
            batch_df.to_csv(output_csv, mode='a', index=False, header=False, encoding="utf-8-sig")
        print(f"已處理 {end_idx} 筆 / {total}")
        time.sleep(1)

    print("全部處理完成。最終結果已寫入：", output_csv)

if __name__ == "__main__":
    main()