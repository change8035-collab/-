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
    """Madmom(딥러닝) + librosa로 정확한 비트/온셋 기반 노트맵 생성"""
    try:
        import numpy as np
        import subprocess
        import tempfile
        print(f'[분석 시작] {filepath}')

        # webm/ogg 등은 wav로 변환
        wav_path = filepath
        tmp_wav = None
        if filepath.endswith(('.webm', '.ogg', '.m4a', '.opus', '.mp3')):
            try:
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except:
                ffmpeg_bin = 'ffmpeg'
            tmp_wav = tempfile.mktemp(suffix='.wav')
            result = subprocess.run([ffmpeg_bin, '-i', filepath, '-ar', '44100', '-ac', '1', '-f', 'wav', tmp_wav, '-y'],
                                    capture_output=True, timeout=60)
            if result.returncode == 0:
                wav_path = tmp_wav
                print(f'[변환 완료] {filepath} → wav')
            else:
                print(f'[변환 실패] {result.stderr.decode()[:200]}')

        # librosa로 기본 정보 로드
        import librosa
        y, sr = librosa.load(wav_path, sr=44100)
        duration = librosa.get_duration(y=y, sr=sr)

        # ── Madmom 비트 감지 (딥러닝 기반, 매우 정확) ──
        beat_times = None
        bpm = 120
        try:
            import madmom
            # RNNBeatProcessor: 딥러닝으로 비트 확률 계산
            proc = madmom.features.beats.RNNBeatProcessor()
            act = proc(wav_path)
            # DBNBeatTrackingProcessor: 동적 베이지안 네트워크로 비트 위치 확정
            beat_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            beat_times = beat_proc(act)
            print(f'[Madmom] 비트 {len(beat_times)}개 감지')

            # BPM 계산 (비트 간격으로)
            if len(beat_times) > 2:
                intervals = np.diff(beat_times)
                median_interval = np.median(intervals)
                if median_interval > 0:
                    bpm = int(round(60.0 / median_interval))
                    bpm = max(60, min(300, bpm))
        except Exception as e:
            print(f'[Madmom 실패, librosa 폴백] {e}')
            # librosa 폴백
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            if hasattr(tempo, '__len__'):
                tempo = tempo[0]
            bpm = int(round(float(tempo)))
            bpm = max(60, min(300, bpm))
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # ── 주파수 대역별 onset 감지 (레인 배분용) ──
        # 저음 (킥/베이스) → 레인 0
        y_low = librosa.effects.preemphasis(y, coef=-0.97)  # 저음 강조
        onset_low = librosa.onset.onset_detect(y=y_low, sr=sr, units='time',
                                                 hop_length=512, backtrack=True,
                                                 pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.15)

        # 고음 (하이햇/심벌) → 레인 3
        y_high = librosa.effects.preemphasis(y, coef=0.97)  # 고음 강조
        onset_high = librosa.onset.onset_detect(y=y_high, sr=sr, units='time',
                                                  hop_length=512, backtrack=True,
                                                  pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.15)

        # 스펙트럼 중심 + RMS (레인 미세 조정용)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sc_min, sc_max = spec_cent.min(), spec_cent.max()
        if sc_max > sc_min:
            spec_norm = (spec_cent - sc_min) / (sc_max - sc_min)
        else:
            spec_norm = np.zeros_like(spec_cent)

        rms = librosa.feature.rms(y=y)[0]
        rms_max = rms.max() if rms.max() > 0 else 1
        rms_norm = rms / rms_max

        # ── 비트 + onset 합산 (빠른 곡도 정확하게) ──
        # librosa onset도 감지 (비트 사이 빠른 음 포착)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames_all = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                                       backtrack=True, delta=0.07)
        onset_times_all = librosa.frames_to_time(onset_frames_all, sr=sr)

        # Madmom 비트 + librosa onset 합치기 (중복 제거)
        all_times = set()
        for t in beat_times:
            all_times.add(round(float(t), 3))
        for t in onset_times_all:
            all_times.add(round(float(t), 3))
        # 너무 가까운 노트 제거 (최소 0.08초 간격)
        all_times_sorted = sorted(all_times)
        merged_times = []
        for t in all_times_sorted:
            if len(merged_times) == 0 or t - merged_times[-1] >= 0.08:
                merged_times.append(t)

        beat_set = set(round(float(t), 2) for t in beat_times)
        low_set = set(round(t, 2) for t in onset_low)
        high_set = set(round(t, 2) for t in onset_high)

        notes = []
        prev_lane = -1
        consecutive = 0

        for t in merged_times:
            if t < 0.5 or t > duration - 0.5:
                continue

            t_round = round(float(t), 2)
            frame = librosa.time_to_frames(t, sr=sr, hop_length=512)
            frame = min(frame, len(spec_norm) - 1)

            sc_val = float(spec_norm[frame]) if frame < len(spec_norm) else 0.5
            en_val = float(rms_norm[min(frame, len(rms_norm)-1)])

            # 주파수 대역 기반 레인 배정
            is_low = any(abs(t_round - lt) < 0.05 for lt in low_set)
            is_high = any(abs(t_round - ht) < 0.05 for ht in high_set)

            if is_low and not is_high:
                lane = 0 if sc_val < 0.5 else 1  # 저음 → 왼쪽
            elif is_high and not is_low:
                lane = 3 if sc_val > 0.5 else 2  # 고음 → 오른쪽
            else:
                # 스펙트럼 기반 배정
                if sc_val < 0.25:
                    lane = 0
                elif sc_val < 0.5:
                    lane = 1
                elif sc_val < 0.75:
                    lane = 2
                else:
                    lane = 3

            # 같은 레인 3회 이상 연속 방지
            if lane == prev_lane:
                consecutive += 1
                if consecutive >= 2:
                    lane = (lane + 1 + int(sc_val * 10)) % 4
                    consecutive = 0
            else:
                consecutive = 0
            prev_lane = lane

            notes.append({
                't': round(float(t), 3),
                'l': lane,
                'e': round(en_val, 2)
            })

        # 임시 wav 파일 삭제
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

        print(f'[분석 완료] BPM={bpm}, 노트={len(notes)}개, 길이={round(duration)}초 (Madmom)')
        return {
            'bpm': bpm,
            'duration': round(duration, 1),
            'notes': notes
        }
    except Exception as e:
        print(f'오디오 분석 실패: {e}')
        import traceback
        traceback.print_exc()
        # 임시 파일 정리
        if 'tmp_wav' in dir() and tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)
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

    download_ok = False
    audio_filename = f'{video_id}.webm'
    duration = 0
    title = ''

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get('duration', 0)
            title = info.get('title', 'Unknown')
            name = name or title
            dl_ext = info.get('ext', 'webm')
            audio_filename = f'{video_id}.{dl_ext}'
            download_ok = True
    except Exception as e:
        # 다운로드 실패해도 곡 추가 가능 (유튜브 iframe으로 재생)
        print(f'yt-dlp 다운로드 실패 (iframe으로 대체): {e}')
        # yt-dlp 없이 영상 정보만 가져오기
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'noplaylist': True, 'skip_download': True}) as ydl2:
                info = ydl2.extract_info(url, download=False)
                duration = info.get('duration', 0)
                title = info.get('title', 'Unknown')
                name = name or title
        except:
            pass

    if not name:
        name = f'YouTube #{video_id[:6]}'
    if not duration or duration < 5:
        duration = 180  # 기본 3분

    # 오디오 분석 (BPM + 노트맵) - 다운로드 성공 시만
    import json
    analyzed_json = None
    if download_ok:
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        analysis = analyze_audio(audio_path)
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
