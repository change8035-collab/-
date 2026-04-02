from flask import Flask, render_template, request, jsonify, send_from_directory
import sqlite3
import os
import uuid
import re
import time

# ffmpeg 경로 설정 (imageio-ffmpeg 사용)
try:
    import imageio_ffmpeg
    ffmpeg_path = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ['PATH'] = ffmpeg_path + ':' + os.environ.get('PATH', '')
except ImportError:
    pass

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beatzone.db')
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'audio')

# ── Database ──────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            bpm INTEGER NOT NULL,
            duration REAL NOT NULL,
            source TEXT NOT NULL,
            youtube_id TEXT,
            audio_filename TEXT NOT NULL,
            analyzed_notes TEXT,
            color TEXT DEFAULT '#ff4444',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id INTEGER NOT NULL,
            difficulty TEXT NOT NULL,
            player_id TEXT NOT NULL,
            nickname TEXT NOT NULL,
            score INTEGER NOT NULL,
            accuracy INTEGER NOT NULL,
            grade TEXT NOT NULL,
            max_combo INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (song_id) REFERENCES songs(id)
        );
    ''')
    conn.commit()
    conn.close()

# ── Helpers ───────────────────────────────────────

def parse_youtube_id(url):
    patterns = [
        r'(?:v=|\/v\/|youtu\.be\/|embed\/)([a-zA-Z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def analyze_audio(filepath):
    """오디오 파일에서 BPM + 비트/음높이 기반 노트맵 생성"""
    try:
        import librosa
        import numpy as np
        import subprocess
        import tempfile
        print(f'[분석 시작] {filepath}')

        # webm/ogg 등은 librosa가 직접 못 읽으므로 wav로 변환
        wav_path = filepath
        tmp_wav = None
        if filepath.endswith(('.webm', '.ogg', '.m4a', '.opus')):
            try:
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except:
                ffmpeg_bin = 'ffmpeg'
            tmp_wav = tempfile.mktemp(suffix='.wav')
            result = subprocess.run([ffmpeg_bin, '-i', filepath, '-ar', '22050', '-ac', '1', '-f', 'wav', tmp_wav, '-y'],
                                    capture_output=True, timeout=60)
            if result.returncode == 0:
                wav_path = tmp_wav
                print(f'[변환 완료] {filepath} → wav')
            else:
                print(f'[변환 실패] {result.stderr.decode()[:200]}')

        y, sr = librosa.load(wav_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)

        # 임시 wav 파일 삭제
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

        # BPM 감지
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # tempo가 배열인 경우 처리
        if hasattr(tempo, '__len__'):
            tempo = tempo[0]
        bpm = int(round(float(tempo)))
        bpm = max(60, min(300, bpm))

        # 온셋 감지 (실제 음이 시작되는 지점)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # 스펙트럼 중심 (음 높낮이 대용)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        # 정규화 (0~1)
        sc_min, sc_max = spec_cent.min(), spec_cent.max()
        if sc_max > sc_min:
            spec_norm = (spec_cent - sc_min) / (sc_max - sc_min)
        else:
            spec_norm = np.zeros_like(spec_cent)

        # RMS 에너지
        rms = librosa.feature.rms(y=y)[0]
        rms_max = rms.max() if rms.max() > 0 else 1
        rms_norm = rms / rms_max

        # 노트 생성: 온셋마다 음높이→레인, 에너지→강도
        # 스펙트럼 값을 percentile 기반으로 4등분 (고른 분배)
        onset_sc_vals = []
        onset_frames_valid = []
        onset_times_valid = []
        onset_en_vals = []
        for i, t in enumerate(onset_times):
            if t < 1.0:
                continue
            frame = onset_frames[i] if i < len(onset_frames) else len(spec_norm) - 1
            if frame >= len(spec_norm):
                frame = len(spec_norm) - 1
            sc_val = float(spec_norm[frame])
            en_val = float(rms_norm[min(frame, len(rms_norm)-1)])
            onset_sc_vals.append(sc_val)
            onset_frames_valid.append(frame)
            onset_times_valid.append(t)
            onset_en_vals.append(en_val)

        # percentile 기반 4분위 (각 레인에 25%씩 배분)
        if len(onset_sc_vals) > 0:
            sc_arr = np.array(onset_sc_vals)
            q25 = np.percentile(sc_arr, 25)
            q50 = np.percentile(sc_arr, 50)
            q75 = np.percentile(sc_arr, 75)
        else:
            q25 = q50 = q75 = 0.5

        notes = []
        prev_lane = -1
        for i in range(len(onset_times_valid)):
            sc_val = onset_sc_vals[i]
            en_val = onset_en_vals[i]
            t = onset_times_valid[i]

            # percentile 기반 레인 배정 (고르게)
            if sc_val < q25:
                lane = 0
            elif sc_val < q50:
                lane = 1
            elif sc_val < q75:
                lane = 2
            else:
                lane = 3

            # 같은 레인이 3번 이상 연속되면 인접 레인으로 이동
            if lane == prev_lane:
                shift = 1 if lane < 2 else -1
                lane = max(0, min(3, lane + shift))
            prev_lane = lane

            notes.append({
                't': round(float(t), 3),
                'l': lane,
                'e': round(en_val, 2)
            })

        print(f'[분석 완료] BPM={bpm}, 노트={len(notes)}개, 길이={round(duration)}초')
        return {
            'bpm': bpm,
            'duration': round(duration, 1),
            'notes': notes
        }
    except Exception as e:
        print(f'오디오 분석 실패: {e}')
        import traceback
        traceback.print_exc()
        return None

def song_to_dict(row):
    import json
    notes = None
    if row['analyzed_notes']:
        try:
            notes = json.loads(row['analyzed_notes'])
        except:
            pass
    return {
        'id': row['id'],
        'name': row['name'],
        'bpm': row['bpm'],
        'duration': row['duration'],
        'source': row['source'],
        'youtube_id': row['youtube_id'],
        'audio_url': '/static/audio/' + row['audio_filename'],
        'color': row['color'] or '#ff4444',
        'analyzed_notes': notes,
    }

# ── Routes ────────────────────────────────────────

@app.route('/')
def index():
    return render_template('game.html')

# ── Songs API ─────────────────────────────────────

@app.route('/api/songs')
def list_songs():
    conn = get_db()
    rows = conn.execute('SELECT * FROM songs ORDER BY created_at DESC').fetchall()
    conn.close()
    return jsonify([song_to_dict(r) for r in rows])

@app.route('/api/songs/youtube', methods=['POST'])
def add_youtube_song():
    data = request.get_json()
    url = data.get('url', '').strip()
    bpm = int(data.get('bpm', 120))
    name = data.get('name', '').strip()

    video_id = parse_youtube_id(url)
    if not video_id:
        return jsonify({'error': '올바른 YouTube URL이 아닙니다'}), 400

    # 이미 다운로드된 곡인지 확인
    conn = get_db()
    existing = conn.execute('SELECT * FROM songs WHERE youtube_id=?', (video_id,)).fetchone()
    if existing:
        conn.close()
        return jsonify({'error': '이미 추가된 곡입니다', 'song': song_to_dict(existing)}), 409

    # yt-dlp로 오디오 다운로드
    try:
        import yt_dlp
    except ImportError:
        conn.close()
        return jsonify({'error': 'yt-dlp가 설치되지 않았습니다. pip install yt-dlp'}), 500

    audio_filename = f'{video_id}.mp3'
    output_path = os.path.join(AUDIO_DIR, video_id)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path + '.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'no_color': True,
        'noprogress': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get('duration', 0)
            title = info.get('title', 'Unknown')
            name = name or title
            dl_ext = info.get('ext', 'webm')
            audio_filename = f'{video_id}.{dl_ext}'
    except Exception as e:
        conn.close()
        return jsonify({'error': f'다운로드 실패: {str(e)}'}), 500

    if not duration or duration < 5:
        duration = 90

    # 오디오 분석 (BPM + 노트맵)
    import json
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    analysis = analyze_audio(audio_path)
    analyzed_json = None
    if analysis:
        bpm = analysis['bpm']
        duration = analysis['duration']
        analyzed_json = json.dumps(analysis['notes'])

    # DB에 저장
    bpm = max(60, min(300, bpm))
    conn.execute(
        'INSERT INTO songs (name, bpm, duration, source, youtube_id, audio_filename, analyzed_notes) VALUES (?,?,?,?,?,?,?)',
        (name, bpm, duration, 'youtube', video_id, audio_filename, analyzed_json)
    )
    conn.commit()

    row = conn.execute('SELECT * FROM songs WHERE youtube_id=?', (video_id,)).fetchone()
    conn.close()

    return jsonify(song_to_dict(row)), 201

@app.route('/api/songs/upload', methods=['POST'])
def upload_song():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': '파일이 선택되지 않았습니다'}), 400

    bpm = int(request.form.get('bpm', 120))
    name = request.form.get('name', '').strip() or os.path.splitext(file.filename)[0]
    duration = float(request.form.get('duration', 0))

    # 파일 저장
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.mp3', '.wav', '.ogg', '.m4a', '.webm'):
        return jsonify({'error': '지원하지 않는 파일 형식입니다 (mp3, wav, ogg, m4a)'}), 400

    uid = uuid.uuid4().hex[:12]
    audio_filename = f'upload_{uid}{ext}'
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    file.save(audio_path)

    if duration < 5:
        duration = 90

    # 오디오 분석
    import json
    analysis = analyze_audio(audio_path)
    analyzed_json = None
    if analysis:
        bpm = analysis['bpm']
        duration = analysis['duration']
        analyzed_json = json.dumps(analysis['notes'])

    bpm = max(60, min(300, bpm))
    conn = get_db()
    conn.execute(
        'INSERT INTO songs (name, bpm, duration, source, audio_filename, analyzed_notes) VALUES (?,?,?,?,?,?)',
        (name, bpm, duration, 'upload', audio_filename, analyzed_json)
    )
    conn.commit()

    row = conn.execute('SELECT * FROM songs ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()

    return jsonify(song_to_dict(row)), 201

@app.route('/api/songs/<int:song_id>', methods=['DELETE'])
def delete_song(song_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM songs WHERE id=?', (song_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': '곡을 찾을 수 없습니다'}), 404

    # 오디오 파일 삭제
    audio_path = os.path.join(AUDIO_DIR, row['audio_filename'])
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # DB에서 삭제 (랭킹도 함께)
    conn.execute('DELETE FROM rankings WHERE song_id=?', (song_id,))
    conn.execute('DELETE FROM songs WHERE id=?', (song_id,))
    conn.commit()
    conn.close()

    return jsonify({'ok': True})

@app.route('/api/songs/<int:song_id>/reanalyze', methods=['POST'])
def reanalyze_song(song_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM songs WHERE id=?', (song_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': '곡 없음'}), 404
    audio_path = os.path.join(AUDIO_DIR, row['audio_filename'])
    if not os.path.exists(audio_path):
        conn.close()
        return jsonify({'error': '오디오 파일 없음'}), 404
    import json
    analysis = analyze_audio(audio_path)
    if analysis:
        conn.execute('UPDATE songs SET bpm=?, duration=?, analyzed_notes=? WHERE id=?',
                      (analysis['bpm'], analysis['duration'], json.dumps(analysis['notes']), song_id))
        conn.commit()
    conn.close()
    return jsonify({'ok': True, 'notes': len(analysis['notes']) if analysis else 0})

@app.route('/api/songs/<int:song_id>', methods=['PUT'])
def update_song(song_id):
    data = request.get_json()
    conn = get_db()
    row = conn.execute('SELECT * FROM songs WHERE id=?', (song_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': '곡을 찾을 수 없습니다'}), 404

    new_name = data.get('name', row['name'])
    new_url = data.get('youtube_url')

    if new_url and new_url != row['youtube_url']:
        # URL 변경 시: 기존 오디오 삭제 → 새로 다운로드
        vid = parse_youtube_id(new_url)
        if not vid:
            conn.close()
            return jsonify({'error': '올바른 YouTube URL이 아닙니다'}), 400

        # 기존 파일 삭제
        old_audio = os.path.join(AUDIO_DIR, row['audio_filename'])
        if os.path.exists(old_audio):
            os.remove(old_audio)

        # 새 다운로드
        outfile = os.path.join(AUDIO_DIR, vid + '.webm')
        ffmpeg_path = get_ffmpeg_path()
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outfile,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        if ffmpeg_path:
            ydl_opts['ffmpeg_location'] = ffmpeg_path

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(new_url, download=True)
                duration = info.get('duration', 90)
                title = info.get('title', new_name)
            # 실제 저장된 파일명 확인
            actual_file = None
            for f in os.listdir(AUDIO_DIR):
                if f.startswith(vid):
                    actual_file = f
                    break
            if not actual_file:
                conn.close()
                return jsonify({'error': '다운로드 실패'}), 500

            # BPM 분석
            analyzed = analyze_audio(os.path.join(AUDIO_DIR, actual_file))
            bpm = analyzed['bpm'] if analyzed else row['bpm']
            notes_json = json.dumps(analyzed['notes']) if analyzed else row['analyzed_notes']

            conn.execute('''UPDATE songs SET name=?, youtube_url=?, youtube_id=?,
                audio_filename=?, duration=?, bpm=?, analyzed_notes=? WHERE id=?''',
                (new_name, new_url, vid, actual_file, duration, bpm, notes_json, song_id))
        except Exception as e:
            conn.close()
            return jsonify({'error': f'다운로드 실패: {str(e)}'}), 500
    else:
        # 이름만 변경
        conn.execute('UPDATE songs SET name=? WHERE id=?', (new_name, song_id))

    conn.commit()
    updated = conn.execute('SELECT * FROM songs WHERE id=?', (song_id,)).fetchone()
    conn.close()
    return jsonify(song_to_dict(updated))

# ── Rankings API ──────────────────────────────────

@app.route('/api/rankings/<int:song_id>/<difficulty>')
def get_rankings(song_id, difficulty):
    conn = get_db()
    rows = conn.execute(
        'SELECT nickname, score, accuracy, grade, max_combo, created_at '
        'FROM rankings WHERE song_id=? AND difficulty=? '
        'ORDER BY score DESC LIMIT 10',
        (song_id, difficulty)
    ).fetchall()
    conn.close()

    return jsonify([{
        'nick': r['nickname'],
        'score': r['score'],
        'acc': r['accuracy'],
        'grade': r['grade'],
        'combo': r['max_combo'],
        'date': r['created_at'],
    } for r in rows])

@app.route('/api/rankings', methods=['POST'])
def submit_score():
    data = request.get_json()
    required = ['song_id', 'difficulty', 'nickname', 'score', 'accuracy', 'grade', 'max_combo', 'player_id']
    for key in required:
        if key not in data:
            return jsonify({'error': f'{key} 필드가 없습니다'}), 400

    conn = get_db()
    # player_id 기준으로 같은 곡+난이도 기록 확인
    existing = conn.execute(
        'SELECT id, score FROM rankings WHERE song_id=? AND difficulty=? AND player_id=?',
        (data['song_id'], data['difficulty'], data['player_id'])
    ).fetchone()

    if existing:
        # 기존 점수보다 높을 때만 갱신 (닉네임도 항상 최신으로)
        if data['score'] > existing['score']:
            conn.execute(
                'UPDATE rankings SET score=?, accuracy=?, grade=?, max_combo=?, nickname=?, created_at=CURRENT_TIMESTAMP '
                'WHERE id=?',
                (data['score'], data['accuracy'], data['grade'], data['max_combo'], data['nickname'], existing['id'])
            )
        else:
            # 점수는 안 바뀌어도 닉네임은 갱신
            conn.execute('UPDATE rankings SET nickname=? WHERE id=?', (data['nickname'], existing['id']))
    else:
        conn.execute(
            'INSERT INTO rankings (song_id, difficulty, player_id, nickname, score, accuracy, grade, max_combo) '
            'VALUES (?,?,?,?,?,?,?,?)',
            (data['song_id'], data['difficulty'], data['player_id'], data['nickname'],
             data['score'], data['accuracy'], data['grade'], data['max_combo'])
        )

    conn.commit()
    conn.close()

    return jsonify({'ok': True}), 201

@app.route('/api/nickname', methods=['POST'])
def update_nickname():
    data = request.get_json()
    player_id = data.get('player_id')
    new_nick = data.get('nickname')
    if not player_id or not new_nick:
        return jsonify({'error': 'player_id, nickname 필요'}), 400
    conn = get_db()
    conn.execute('UPDATE rankings SET nickname=? WHERE player_id=?', (new_nick, player_id))
    conn.commit()
    conn.close()
    return jsonify({'ok': True})

# ── Main ──────────────────────────────────────────

def analyze_unprocessed_songs():
    """분석 안 된 곡들 자동 분석"""
    import json
    conn = get_db()
    rows = conn.execute('SELECT id, audio_filename FROM songs WHERE analyzed_notes IS NULL').fetchall()
    for row in rows:
        filepath = os.path.join(AUDIO_DIR, row['audio_filename'])
        if os.path.exists(filepath):
            print(f'[자동 분석] Song ID={row["id"]}: {row["audio_filename"]}')
            result = analyze_audio(filepath)
            if result:
                conn.execute('UPDATE songs SET analyzed_notes=?, bpm=?, duration=? WHERE id=?',
                    (json.dumps(result['notes']), result['bpm'], result['duration'], row['id']))
                conn.commit()
                print(f'[자동 분석 완료] Song ID={row["id"]}: {len(result["notes"])}개 노트, BPM={result["bpm"]}')
    conn.close()

if __name__ == '__main__':
    os.makedirs(AUDIO_DIR, exist_ok=True)
    init_db()
    analyze_unprocessed_songs()
    app.run(debug=True, port=7000)
