# ============================================
# AI Meeting Intelligence Full Research Pipeline
# With Integrated PDF Report Generation
# With Performance Evaluation (cProfile + time)
# ============================================

# -------------------------------
# Imports
# -------------------------------
import whisper
import torchaudio
from pyannote.audio import Pipeline
import spacy
import yake
import pandas as pd
import re
from datetime import datetime
import time
import cProfile
import pstats

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet


# -------------------------------
# Fix torchaudio compatibility
# -------------------------------
if not hasattr(torchaudio, "set_audio_backend"):
    def _noop_backend(*args, **kwargs):
        pass
    torchaudio.set_audio_backend = _noop_backend


# -------------------------------
# MAIN PIPELINE FUNCTION
# -------------------------------
def run_pipeline():

    # -------------------------------
    # Load Models
    # -------------------------------
    print("Loading Whisper...")
    whisper_model = whisper.load_model("base", device="cpu")

    print("Loading Speaker Diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=True
    )

    print("Loading NLP Model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading Keyword Extractor...")
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=5)


    # -------------------------------
    # Action Item Rules
    # -------------------------------
    action_patterns = [
        #Direct tasks AV.
    r'\bwill\b',
    r'\bshall\b',
    r'\bmust\b',
    r'\bneed\s+to\b',
    r'\bhave\s+to\b',
    r'\brequired\s+to\b',
    r'\bresponsiable\s+for\b',
    r'\bassigned\s+to\b',
    r'\bshould\b',
    r'\bplan\s+to\b',
    r'\bhave\s+to\b',

    #Planning and Future Work AV

    r'\bplan\s+to\b',
    r'\bgoing\s+to\b',
    r'\bscheduled\s+to\b',
    r'\bexpected\s+to\b',
    r'\btarget\s+to\b',
    r'\baim\s+to\b',
    r'\bintend\s+to\b',
    
    #Group action AV
    r"let['’]s",
    r'\blet us\b' ,
    r'\bwe\s+should\b',
    r'\bwe\s+must\b',
    r'\bwe\s+need\s+to\b',

    #Decision Indicators AV

    r'\bdecided\s+to\b',
    r'\bagreed\s+to\b',
    r'\bapproved\b',
    r'\bconfirmed\b',
    r'\bfinalized\b',

    #follow up AV

    r'\bfollow\s+up\b',
    r'\bcheck\s+with\b',
    r'\bupdate\s+on\b',
    r'\breview\s+again\b',
    r'\brevisit\b',

    #Reporting tasks AV
    r'\bprepare\s+report\b',
    r'\bsend\s+report\b',
    r'\bsubmit\b',
    r'\bshare\s+update\b',

    #business tasks AV

    r'\bimplement\b',
    r'\bdeploy\b', 
    r'\btest\b',
    r'\bvalidate\b',
    r'\banalyze\b',
    r'\binvestigate\b',
    r'\bdesign\b',
    r'\bdevelop\b',
    r'\bfix\b',
    r'\bresolve\b'
        
    ]


    # -------------------------------
    # Helper Functions
    # -------------------------------
    def detect_priority(text):
        text = text.lower()
        if any(w in text for w in ["urgent", "asap", "immediately", "critical"]):
            return "HIGH"
        elif any(w in text for w in ["important", "required", "need to"]):
            return "MEDIUM"
        else:
            return "LOW"

    def detect_deadline(text):
        text = text.lower()
        patterns = [
            r'by tomorrow', r'next week',
            r'next meeting', r'within \d+ days',
            r'by \w+day', r'before \w+'
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group()
        return "Not Mentioned"

    def split_sentences(text):
        return [s.text.strip() for s in nlp(text).sents]

    def is_action(sentence):
        return any(re.search(p, sentence.lower()) for p in action_patterns)

    def extract_keywords(text):
        return [k for k, _ in kw_extractor.extract_keywords(text)]

    def get_speaker(diarization, start, end):
        max_overlap = 0
        speaker_label = "UNKNOWN"
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > max_overlap:
                max_overlap = overlap
                speaker_label = speaker
        return speaker_label


    # -------------------------------
    # AUDIO PATH
    # -------------------------------
    audio_path = "amicorpus/ES2008a/audio/ES2008a.Mix-Headset.wav"


    # -------------------------------
    # TRANSCRIPTION
    # -------------------------------
    print("Running Speech Recognition...")
    whisper_result = whisper_model.transcribe(audio_path)
    segments = whisper_result["segments"]


    # -------------------------------
    # DIARIZATION
    # -------------------------------
    print("Running Speaker Diarization...")
    diarization = pipeline(audio_path)


    # -------------------------------
    # PROCESSING
    # -------------------------------
    final_data = []

    for seg in segments:
        speaker = get_speaker(diarization, seg["start"], seg["end"])
        sentences = split_sentences(seg["text"])

        for sentence in sentences:
            action_flag = is_action(sentence)

            final_data.append({
                "speaker": speaker,
                "start_time": seg["start"],
                "end_time": seg["end"],
                "sentence": sentence,
                "keywords": extract_keywords(sentence),
                "action_item": action_flag,
                "priority": detect_priority(sentence) if action_flag else "NA",
                "deadline": detect_deadline(sentence) if action_flag else "NA"
            })


    # -------------------------------
    # SAVE CSV
    # -------------------------------
    df = pd.DataFrame(final_data)
    df.to_csv("meeting_full_report.csv", index=False)
    print("CSV Saved")


    # -------------------------------
    # SPEAKER SUMMARY
    # -------------------------------
    speaker_summary = df.groupby("speaker")["sentence"] \
                        .apply(lambda x: " ".join(x)) \
                        .reset_index()


    # -------------------------------
    # PDF REPORT GENERATION
    # -------------------------------
    print("Generating PDF Report...")

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate("Final_Meeting_Report.pdf")
    elements = []

    elements.append(Paragraph("AI Meeting Documentation Report", styles["Title"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Generated On: " + str(datetime.now()), styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Speaker Key Contributions", styles["Heading2"]))

    for _, row in speaker_summary.iterrows():
        elements.append(
            Paragraph(f"<b>{row['speaker']}</b>: {row['sentence'][:500]}", styles["BodyText"])
        )
        elements.append(Spacer(1, 10))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Action Items", styles["Heading2"]))

    action_df = df[df["action_item"] == True]
    table_data = [["Speaker", "Sentence", "Priority", "Deadline"]]

    for _, row in action_df.iterrows():
        table_data.append([
            row["speaker"],
            row["sentence"],
            row["priority"],
            row["deadline"]
        ])

    table = Table(table_data)
    elements.append(table)

    doc.build(elements)

    print("PDF Report Generated Successfully")


# -------------------------------
# PROFILING EXECUTION
# -------------------------------
if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()

    run_pipeline()

    end_time = time.time()
    profiler.disable()

    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats(30)

    # Save profile for visualization
    profiler.dump_stats("profile_results.prof")
    print("\nProfile saved as profile_results.prof")
