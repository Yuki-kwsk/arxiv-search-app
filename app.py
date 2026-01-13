from flask import Flask, request, jsonify, render_template
import arxiv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import re
from collections import Counter

app = Flask(__name__)

# 翻訳モデルを起動時に一度だけ読み込む
# 初回実行時にはモデルのダウンロードが行われるため、少し時間がかかります。
print("モデルを読み込んでいます...")
try:
    # 翻訳モデル
    translation_tokenizer = AutoTokenizer.from_pretrained("staka/fugumt-en-ja")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("staka/fugumt-en-ja")
    print("モデルの読み込みが完了しました。")
except Exception as e:
    print(f"モデルの読み込み中にエラーが発生しました: {e}")

# キーワード抽出用の基本的なストップワードリスト
STOP_WORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it',
    'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so',
    'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these',
    'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'we', 'us', 'our', 'they', 'them', 'their', 'it', 'its', 'this', 'that', 'these', 'those',
    'paper', 'research', 'study', 'results', 'method', 'methods', 'approach', 'based', 'propose', 'proposed',
    'show', 'model', 'models', 'data', 'using', 'also', 'however', 'one', 'two', 'three', 'via'
])

@app.route('/')
def index():
    """ルートURLにアクセスされた際にindex.htmlを返す"""
    return render_template('index.html')

@app.route('/search')
def search():
    """
    GETリクエストで受け取ったクエリに基づいてarXivで論文を検索し、
    結果と関連キーワード候補をJSON形式で返すAPIエンドポイント
    """
    try:
        main_query = request.args.get('query', '')
        max_results = int(request.args.get('max_results', 5))
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        sort_by_param = request.args.get('sort_by', 'relevance') # デフォルトは 'relevance'

        if not main_query:
            return jsonify({"error": "Query parameter is required"}), 400

        # arXiv API用のクエリを組み立てる
        query_parts = [main_query]
        if start_date or end_date:
            # YYYY-MM-DD を YYYYMMDD 形式に変換
            start_date_arxiv = start_date.replace('-', '') if start_date else '*'
            end_date_arxiv = end_date.replace('-', '') if end_date else '*'

            # 期間が指定されている場合、開始時刻と終了時刻を補完
            if start_date_arxiv != '*':
                start_date_arxiv += "0000" # 00:00 JST
            if end_date_arxiv != '*':
                end_date_arxiv += "1459" # 23:59 JST (UTCでの翌日14:59)

            query_parts.append(f'submittedDate:[{start_date_arxiv} TO {end_date_arxiv}]')

        final_query = " AND ".join(query_parts)

        # 並び替え順の決定
        if sort_by_param == 'newest':
            sort_criterion = arxiv.SortCriterion.SubmittedDate
            sort_order = arxiv.SortOrder.Descending
        elif sort_by_param == 'oldest':
            sort_criterion = arxiv.SortCriterion.SubmittedDate
            sort_order = arxiv.SortOrder.Ascending
        else: # relevance
            sort_criterion = arxiv.SortCriterion.Relevance
            sort_order = arxiv.SortOrder.Descending

        client = arxiv.Client()
        # arXiv APIで論文を検索
        search = arxiv.Search(
            query=final_query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=sort_order
        )

        results = []
        all_summaries_text = ""
        for result in client.results(search):
            # BibTeXエントリーを手動で作成
            bibtex_entry = f"""@misc{{{result.get_short_id()},
      author    = {{{' and '.join(author.name for author in result.authors)}}},
      title     = {{{result.title}}},
      year      = {{{result.published.year}}},
      eprint    = {{{result.get_short_id().split('v')[0]}}},
      archivePrefix = {{arXiv}},
      primaryClass = {{{result.primary_category}}}
}}"""
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary.replace('\n', ' '),
                "published": result.published.strftime('%Y-%m-%d'),
                "pdf_url": result.pdf_url,
                "bibtex": bibtex_entry
            }
            results.append(paper)
            all_summaries_text += " " + paper['summary']

        # 関連キーワードの抽出
        words = re.findall(r'\b\w{3,}\b', all_summaries_text.lower())
        filtered_words = [word for word in words if word not in STOP_WORDS and not word.isdigit()]
        word_counts = Counter(filtered_words)
        for query_word in main_query.lower().split():
            if query_word in word_counts:
                del word_counts[query_word]
        suggestions = [word for word, count in word_counts.most_common(10)]

        return jsonify({"papers": results, "suggestions": suggestions})
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    """
    POSTリクエストで受け取った英語のテキストを日本語に翻訳して返すAPIエンドポイント
    """
    data = request.get_json()
    text_to_translate = data.get('text', '')

    if not text_to_translate:
        return jsonify({"error": "翻訳するテキストが必要です"}), 400

    try:
        # テキストを文に分割
        # nltk.sent_tokenizeの代わりに正規表現で文を分割する
        # この正規表現は、ピリオド、疑問符、感嘆符の後ろにスペースが続く箇所で分割を試みます。
        # (例: "Mr. Smith" のような略称では分割しないように、より複雑なパターンも考慮できますが、まずはシンプルな方法で対応します)
        sentences = re.split(r'(?<=[.?!])\s+', text_to_translate)

        # 各文を翻訳
        # 長い要旨の場合、一度にすべてを翻訳しようとするとモデルの制限を超える可能性があるため、
        # 文ごとに分割して翻訳し、最後に結合します。
        translated_sentences = []
        for sentence in sentences:
            if sentence.strip(): # 空の文は無視
                # 翻訳の実行
                inputs = translation_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
                outputs = translation_model.generate(
                    **inputs,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                decoded_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated_sentences.append(decoded_text)

        # 翻訳された文を結合
        return jsonify({"translation": " ".join(translated_sentences)})
    except Exception as e:
        print(f"--- 翻訳中にエラーが発生しました ---")
        print(f"エラー内容: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 本番環境ではGunicornなどのWSGIサーバーから実行されるため、このブロックはローカルでの開発時のみ実行されます。
    # HerokuはPORT環境変数を自動で設定するため、portの指定はローカル開発用です。
    app.run(debug=True, host='0.0.0.0', port=8080)
