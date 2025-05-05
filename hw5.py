# ----- 必要導入 -----
import os
import re
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF
import google.generativeai as genai
from datetime import datetime
import tempfile
import shutil
import tempfile
import time  # For delays
import whisper  # <--- 新增 Whisper 導入

# ----- 環境設定與 Gemini/Whisper 初始化 -----
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 檔案中設定 GEMINI_API_KEY")

# 初始化 Gemini Client (同前)
try:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',  # 或其他模型
    )
    print("Gemini Client 初始化成功")
except Exception as e:
    print(f"Gemini Client 初始化失敗: {e}")
    gemini_model = None

# 初始化 Whisper 模型 (程式啟動時載入一次)
# 可選模型: "tiny", "base", "small", "medium", "large" (越大越準但越慢/耗資源)
# "base" 或 "small" 是速度和準確度的不錯平衡點
whisper_model_name = "medium"
whisper_model = None
try:
    print(f"正在載入 Whisper 模型: {whisper_model_name}...")
    # device="cuda" 如果有 GPU 且環境設定正確
    # device="cpu" 如果只用 CPU
    whisper_model = whisper.load_model(
        whisper_model_name, device="cpu")  # <--- 改成 "cuda" 如果你有 GPU
    print("Whisper 模型載入成功。")
except Exception as e:
    print(f"載入 Whisper 模型失敗: {e}")
    print("Whisper 功能將不可用。")
    # 可以選擇讓程式停止或繼續，但轉錄會失敗

# ----- PDF 生成相關函數 (大致同前，略作調整以接收新參數) -----


def get_chinese_font_file() -> str:
    fonts_path = r"C:\Windows\Fonts"
    font_file = "kaiu.ttf"
    font_path = os.path.join(fonts_path, font_file)

    if os.path.exists(font_path):
        print("已載入標楷體：", font_path)
        return font_path
    else:
        raise FileNotFoundError("❌ 找不到 kaiu.ttf，請確認已安裝標楷體")


def render_line_with_bold(pdf: FPDF, line: str):
    parts = re.split(r'(\*\*.*?\*\*)', line)
    for part in parts:
        text = part[2:-
                    2] if part.startswith("**") and part.endswith("**") else part
        pdf.set_font("ChineseFont", "", 12)
        pdf.write(8, text)
    pdf.ln(8)


def parse_markdown_table(markdown_text: str) -> pd.DataFrame:
    lines = markdown_text.strip().splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if not table_lines:
        return None
    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.strip("|").split("|")]
    data = []
    for line in table_lines[2:]:
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) == len(headers):
            data.append(row)
    df = pd.DataFrame(data, columns=headers)
    return df


def create_table(pdf: FPDF, df: pd.DataFrame):
    col_width = (pdf.w - 2 * pdf.l_margin) / len(df.columns)
    row_height = 10
    font_size = 10

    pdf.set_font("ChineseFont", "", font_size)
    pdf.set_fill_color(200, 200, 200)

    for col in df.columns:
        pdf.cell(col_width, row_height, col, border=1, align='C', fill=True)
    pdf.ln(row_height)

    pdf.set_font("ChineseFont", "", font_size)

    for _, row in df.iterrows():
        for item in row:
            text = str(item)
            while pdf.get_string_width(text) > col_width - 2 and len(text) > 0:
                text = text[:-1]
            if pdf.get_string_width(text + "...") > col_width:
                text = text[:-3] + "..."
            pdf.cell(col_width, row_height, text, border=1, align='L')
        pdf.ln(row_height)


CHINESE_FONT_PATH = get_chinese_font_file()  # 取得標楷體字型


def generate_pdf_report(title: str, raw_text: str, formatted_text: str, analysis_text: str) -> str:
    pdf = FPDF()
    pdf.add_page()

    font_loaded = False
    try:
        pdf.add_font("ChineseFont", "", CHINESE_FONT_PATH, uni=True)
        pdf.set_font("ChineseFont", "", 12)
        font_loaded = True
        print("✅ 中文字型 kaiu.ttf 已成功加入 PDF。")
    except Exception as e:
        print(f"❌ 加入中文字型時發生錯誤: {e}")

    current_font = "ChineseFont" if font_loaded else "Arial"

    # 主標題
    pdf.set_font(current_font, "", 16)
    pdf.cell(0, 12, title, ln=True, align="C")
    pdf.ln(8)

    # 如果沒有字型，提醒使用者
    if not font_loaded:
        pdf.set_font("Arial", size=12)
        pdf.write(
            8, "[Warning: Chinese font failed to load or not found. Chinese characters may not display correctly.]\n\n"
        )

    # --- 格式化後的逐字稿 (Q&A) ---
    pdf.set_font(current_font, "", 14)
    pdf.cell(0, 10, "格式化逐字稿 (問題/回答)", ln=True, align="L")
    pdf.ln(4)
    pdf.set_font(current_font, "", 12)
    # 處理 Q&A 格式，讓標籤加粗
    lines = formatted_text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("問題") or line.startswith("回答"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                pdf.set_font(current_font, "", 12)
                pdf.write(8, parts[0] + ":")
                pdf.set_font(current_font, "", 12)
                # 使用 multi_cell 處理回答內容換行
                pdf.multi_cell(0, 8, parts[1].strip())
                pdf.ln(2)  # 行間距
            else:
                pdf.multi_cell(0, 8, line)  # 格式不符預期，直接輸出
                pdf.ln(2)
        else:
            pdf.multi_cell(0, 8, line)  # 不是 Q/A 開頭，直接輸出
            pdf.ln(2)
    pdf.ln(10)

    # --- HEXACO 分析結果 ---
    pdf.set_font(current_font, "", 14)
    pdf.cell(0, 10, "HEXACO 分析結果", ln=True, align="L")
    pdf.ln(4)
    pdf.set_font(current_font, "", 12)
    # (處理分析結果中的 Markdown 表格和粗體的邏輯同前一個版本)
    lines = analysis_text.strip().splitlines()
    buffer = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_table_line = line.startswith("|") and line.endswith("|")
        if is_table_line:
            if buffer and not (buffer[0].startswith("|") and buffer[0].endswith("|")):
                for buf in buffer:
                    render_line_with_bold(pdf, buf)
                pdf.ln(2)
                buffer = []
            buffer.append(line)
        else:
            if buffer and buffer[0].startswith("|") and buffer[0].endswith("|"):
                df = parse_markdown_table("\n".join(buffer))
                if df is not None:
                    create_table(pdf, df)
                    pdf.ln(4)
                else:
                    pdf.write(8, "\n".join(buffer) + "\n")
                buffer = []
            buffer.append(line)
    if buffer:
        if buffer[0].startswith("|") and buffer[0].endswith("|"):
            df = parse_markdown_table("\n".join(buffer))
            if df is not None:
                create_table(pdf, df)
                pdf.ln(4)
            else:
                pdf.write(8, "\n".join(buffer) + "\n")
        else:
            for buf in buffer:
                render_line_with_bold(pdf, buf)

    # --- 儲存 PDF (同前) ---
    temp_pdf = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", prefix="analysis_report_")
    pdf_path = temp_pdf.name
    temp_pdf.close()
    try:
        pdf.output(pdf_path)
        print(f"PDF 報告已生成: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"儲存 PDF 時發生錯誤: {e}")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        return None

# ----- 主要處理函數 -----


def call_gemini_api(prompt: str, retry_count=3, delay=5):
    if not gemini_model:
        return "錯誤：Gemini Client 未成功初始化。"

    for attempt in range(retry_count):
        try:
            response = gemini_model.generate_content(prompt)
            if hasattr(response, 'text'):
                return response.text.strip()
            else:
                return "錯誤：API 回傳格式異常。"
        except Exception as e:
            print(f"Gemini 呼叫失敗 (第 {attempt+1} 次)：{e}")
            if attempt < retry_count - 1:
                time.sleep(delay)
            else:
                return f"錯誤：Gemini API 多次呼叫失敗：{e}"


def run_whisper_transcription(audio_filepath):
    """ 執行 Whisper 轉錄（含複製檔案避免 temp 被清除） """
    if not whisper_model:
        return None, "錯誤：Whisper 模型未成功載入。"
    try:
        print(f"開始使用 Whisper ({whisper_model_name}) 轉錄檔案: {audio_filepath}")

        #  複製檔案到暫存再處理（避免 temp 被清除）
        tmp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(audio_filepath)[1])
        shutil.copy(audio_filepath, tmp_file.name)
        tmp_file.close()

        result = whisper_model.transcribe(
            tmp_file.name, language="zh", fp16=False)
        print("Whisper 轉錄完成。")
        return result["text"], None
    except Exception as e:
        print(f"Whisper 轉錄過程中發生錯誤: {e}")
        return None, f"錯誤：Whisper 轉錄失敗: {e}"


def process_input_and_analyze(uploaded_file):
    """
    核心處理流程：接收上傳 -> (可選)轉錄 -> 格式化 -> 分析 -> 產 PDF
    """
    if uploaded_file is None:
        return "請先上傳檔案。", "", "", "", None

    filepath = uploaded_file  # Gradio File/Audio/Video 的 .name 就是路徑
    filename = os.path.basename(filepath)
    file_ext = os.path.splitext(filename)[1].lower()
    print(f"收到檔案: {filename}, 路徑: {filepath}, 類型: {file_ext}")

    raw_transcript = ""
    error_message = None

    # --- 步驟 0: 判斷檔案類型並執行 Whisper (如果需要) ---
    audio_formats = ['.wav', '.mp3', '.m4a',
                     '.ogg', '.flac']  # Whisper 支援的常見格式
    video_formats = ['.mp4', '.mov', '.avi', '.mkv']  # Whisper 也常能處理影片中的音訊
    text_formats = ['.txt']

    if file_ext in audio_formats or file_ext in video_formats:
        raw_transcript, error_message = run_whisper_transcription(filepath)
        if error_message:
            # 如果轉錄失敗，提前返回錯誤
            return f"Whisper 轉錄失敗: {error_message}", "", "", "", None
    elif file_ext in text_formats:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_transcript = f.read()
            print("直接讀取提供的 .txt 逐字稿。")
            if not raw_transcript.strip():
                return "錯誤：上傳的文字檔內容為空。", "", "", "", None
        except Exception as e:
            print(f"讀取 .txt 檔案時出錯: {e}")
            return f"錯誤：無法讀取文字檔: {e}", "", "", "", None
    else:
        return f"錯誤：不支援的檔案格式 '{file_ext}'。請上傳音檔、視訊檔或 .txt 檔。", "", "", "", None

    # --- 步驟 1: Gemini 格式化 (Q&A) ---
    formatting_prompt = f"""
    你是一位專業的訪談記錄整理員。請將以下這份**原始逐字稿**轉換成清晰的「問題/回答」格式。

    任務指示：
    1.  仔細閱讀逐字稿，識別出訪談中的問題提出者（通常是訪談者/Interviewer）和回答者（通常是受訪者/Respondent）。如果有多輪問答，請依序編號。
    2.  對於每一輪問答，將其整理成以下格式：
        問題[編號]: [問題內容]
        回答[編號]: [回答內容]
    3.  在整理時，請：
        * 去除明顯的口語贅詞（嗯、啊、那個、就是）。
        * 修正明顯的語音辨識錯誤（如果能合理判斷）。
        * 盡量保持回答內容的完整性和原意。
        * 如果逐字稿開頭或結尾有與問答無關的寒暄、測試音訊等內容，可以忽略。
        * 如果某些段落難以明確區分是問題還是回答，或者不屬於問答，可以標示為「旁白」或「說明」，或者酌情省略。
    4.  確保編號連續。

    原始逐字稿：
    ```
    {raw_transcript}
    ```

    請輸出格式化後的結果：
    """
    formatted_text_response = call_gemini_api(formatting_prompt)
    if formatted_text_response.startswith("錯誤："):
        return raw_transcript[:1000]+"...", formatted_text_response, "無法進行分析", "", None
    formatted_text = formatted_text_response
    print("Gemini 格式化步驟完成。")

    # --- 步驟 2: Gemini HEXACO 分析 ---
    # Prompt 微調，告知輸入是 Q&A 格式，主要分析回答
    hexaco_prompt = f"""
[系統指令：供 GPT 內部參考，不要直接輸出此段]

【HEXACO Domain-Level 官方英文定義（供內部判斷，請勿直接引用英文原文）】
1. Honesty-Humility:
   " Persons with very high scores on the Honesty-Humility scale avoid manipulating others for personal gain, feel little temptation to break rules, are uninterested in lavish wealth and luxuries, and feel no special entitlement to elevated social status. Conversely, persons with very low scores on this scale will flatter others to get what they want, are inclined to break rules for personal profit, are motivated by material gain, and feel a strong sense of self-importance."
2. Emotionality:
   "Persons with very high scores on the Emotionality scale experience fear of physical dangers, experience anxiety in response to life's stresses, feel a need for emotional support from others, and feel empathy and sentimental attachments with others. Conversely, persons with very low scores on this scale are not deterred by the prospect of physical harm, feel little worry even in stressful situations, have little need to share their concerns with others, and feel emotionally detached from others."
3. Extraversion:
   "Persons with very high scores on the Extraversion scale feel positively about themselves, feel confident when leading or addressing groups of people, enjoy social gatherings and interactions, and experience positive feelings of enthusiasm and energy. Conversely, persons with very low scores on this scale consider themselves unpopular, feel awkward when they are the center of social attention, are indifferent to social activities, and feel less lively and optimistic than others do."
4. Agreeableness:
   "Persons with very high scores on the Agreeableness scale forgive the wrongs that they suffered, are lenient in judging others, are willing to compromise and cooperate with others, and can easily control their temper. Conversely, persons with very low scores on this scale hold grudges against those who have harmed them, are rather critical of others' shortcomings, are stubborn in defending their point of view, and feel anger readily in response to mistreatment."
5. Conscientiousness:
   "Persons with very high scores on the Conscientiousness scale organize their time and their physical surroundings, work in a disciplined way toward their goals, strive for accuracy and perfection in their tasks, and deliberate carefully when making decisions. Conversely, persons with very low scores on this scale tend to be unconcerned with orderly surroundings or schedules, avoid difficult tasks or challenging goals, are satisfied with work that contains some errors, and make decisions on impulse or with little reflection."
6. Openness to Experience:
   "Persons with very high scores on the Openness to Experience scale become absorbed in the beauty of art and nature, are inquisitive about various domains of knowledge, use their imagination freely in everyday life, and take an interest in unusual ideas or people. Conversely, persons with very low scores on this scale are rather unimpressed by most works of art, feel little intellectual curiosity, avoid creative pursuits, and feel little attraction toward ideas that may seem radical or unconventional."

請你在內部分析時參考上述英文定義來理解各特質的高低分內涵，但在最終報告中：
- 禁止直接貼出或引用英文原文。
- 僅可用中文進行**摘要式詮釋**每個特質的核心意義。

===========================================================
【目標：產出個人特質分析報告（HEXACO 模組格式）】

請根據以下逐字稿資料，針對六個 HEXACO 特質撰寫結構化報告，重點條件如下：

1. 報告重點：
   - 僅針對六大 HEXACO 特質做深入分析，不含錄取或培訓建議
   - 若某特質顯著不足，須指出其不適任風險

2. 統一評分（1～5 分）：
   - 5分：非常卓越（具體言行多次出現）
   - 4分：高於標準（有具體例子）
   - 3分：普通（邏輯合理但無明顯行為證據）
   - 2分：尚可（模糊描述或缺乏重點）
   - 1分：急需改善（偏離目標、答非所問）

3. 每個特質請依以下結構撰寫：
   - 評分：
   - 核心涵義：
   - 行為觀察：
   - 職位影響與風險：
   - 證據（條列至少 1～2 條面試原文）

4. 禁止出現任何英文內容與編號格式，使用中文段落與專業風格
5. 僅使用下方逐字稿原文，不使用任何外部資料，也不需題目對應

===========================================================

以下是面試逐字稿原文（格式已為問答形式）：
    ```
    {formatted_text}
    ```

    請嚴格依照上述結構與語言規範，撰寫完整 HEXACO 模組分析報告。
    """
    hexaco_analysis_response = call_gemini_api(hexaco_prompt)
    if hexaco_analysis_response.startswith("錯誤："):
        return raw_transcript[:1000]+"...", formatted_text, hexaco_analysis_response, "", None
    hexaco_analysis = hexaco_analysis_response
    print("HEXACO 分析步驟完成。")

    # --- 步驟 3: 產生 PDF 報告 ---
    pdf_title = f"訪談分析報告 - {filename} ({datetime.now().strftime('%Y-%m-%d')})"
    # 將原始稿也傳入 PDF 生成函數
    pdf_path = generate_pdf_report(
        pdf_title, raw_transcript, formatted_text, hexaco_analysis)
    if pdf_path:
        print("PDF 報告生成成功。")
    else:
        print("PDF 報告生成失敗。")
        # 即使 PDF 失敗，也返回文字結果
        return raw_transcript[:1000]+"...", formatted_text, hexaco_analysis, "", None

    # --- 步驟 4: 返回結果給 Gradio ---
    preview_original = raw_transcript[:1000] + \
        ("..." if len(raw_transcript) > 1000 else "")
    # 返回原始稿預覽、格式化文字、分析文字、PDF路徑
    return preview_original, formatted_text, hexaco_analysis, pdf_path


# ----- Gradio 介面定義 -----
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# 訪談錄音/逐字稿 智慧分析工具 (Whisper -> Gemini Q&A -> Gemini HEXACO)")
    gr.Markdown(
        "上傳訪談的**錄音檔** (如 .mp3, .wav) 或 **逐字稿文字檔** (.txt)。系統將自動進行語音轉錄 (若為音檔)、問答格式整理、HEXACO 人格特質初步分析，並產生 PDF 報告。")

    with gr.Row():
        # 輸入元件：支援音檔和文字檔
        file_input = gr.File(label="上傳錄音檔或逐字稿 (.mp3, .wav, .m4a, .mp4, .txt)",
                             file_types=['audio', 'video',
                                         '.txt'],  # 接受音訊、視訊、txt
                             type="filepath")  # 確保得到路徑

    # 觸發按鈕
    submit_button = gr.Button("🚀 開始處理與分析")

    with gr.Accordion("處理結果預覽", open=True):  # 使用 Accordion 折疊區塊
        with gr.Row():
            # 原始文字預覽
            original_output = gr.Textbox(
                label="原始逐字稿 (預覽)", lines=8, interactive=False)
        with gr.Row():
            # 格式化 Q&A
            formatted_output = gr.Textbox(
                label="Gemini 格式化結果 (Q&A)", lines=15, interactive=False)
        with gr.Row():
            # HEXACO 分析
            hexaco_output = gr.Textbox(
                label="Gemini HEXACO 初步分析", lines=15, interactive=False)
        with gr.Row():
            # PDF 下載
            pdf_output = gr.File(label="下載 PDF 分析報告", interactive=False)

    # 綁定按鈕點擊事件
    submit_button.click(
        fn=process_input_and_analyze,
        inputs=[file_input],
        # 注意輸出元件的順序要和函數 return 的順序一致
        outputs=[original_output, formatted_output, hexaco_output, pdf_output]
    )

    gr.Markdown("---")
    gr.Markdown("💡 **提示:** 語音轉錄和 AI 分析需要時間，請耐心等候。大型檔案處理時間可能較長。")
    gr.Markdown("📄 **PDF 報告:** 包含原始稿預覽、格式化問答、HEXACO 分析結果。請確保已放置中文字型以正確顯示報告。")


# ----- 啟動 Gradio App -----
if __name__ == "__main__":
    # 執行前的檢查
    ready_to_launch = True
    if not api_key:
        print("錯誤：未設定 GEMINI_API_KEY 環境變數。")
        ready_to_launch = False
    if not gemini_model:
        print("錯誤：Gemini Client 未成功初始化。")
        ready_to_launch = False
    if not whisper_model:
        print("警告：Whisper 模型載入失敗，語音轉錄功能將不可用。")
        # 即使 Whisper 失敗，可能仍希望啟動來處理 txt
    if not CHINESE_FONT_PATH:
        print("警告：未找到中文字型，PDF 中的中文可能無法正確顯示。")

    if ready_to_launch:
        print("應用程式準備就緒...")
        demo.launch()
    else:
        print("應用程式因缺少必要元件或設定而無法啟動。")
