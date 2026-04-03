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
    """Demucs 음원 분리 + librosa onset으로 악기별 정확한 노트맵 생성"""
    try:
        import numpy as np
        import subprocess
        import tempfile
        import shutil
        import librosa
        print(f'[분석 시작] {filepath}')

        # wav 변환
        wav_path = filepath
        tmp_wav = None
        if filepath.endswith(('.webm', '.ogg', '.m4a', '.opus', '.mp3')):
            try:
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except:
                ffmpeg_bin = 'ffmpeg'
            tmp_wav = tempfile.mktemp(suffix='.wav')
            result = subprocess.run([ffmpeg_bin, '-i', filepath, '-ar', '44100', '-ac', '2', '-f', 'wav', tmp_wav, '-y'],
                                    capture_output=True, timeout=60)
            if result.returncode == 0:
                wav_path = tmp_wav
            else:
                print(f'[변환 실패] {result.stderr.decode()[:200]}')

        y_full, sr = librosa.load(wav_path, sr=44100, mono=True)
        duration = librosa.get_duration(y=y_full, sr=sr)

        # ── Demucs 음원 분리 ──
        stems = {}  # drums, bass, vocals, other
        demucs_ok = False
        tmp_demucs_dir = tempfile.mkdtemp()

        try:
            print('[Demucs] 음원 분리 시작...')
            result = subprocess.run(
                ['python3', '-m', 'demucs', '--two-stems=drums', '-n', 'htdemucs',
                 '--out', tmp_demucs_dir, wav_path],
                capture_output=True, timeout=300, text=True
            )
            if result.returncode == 0:
                # Demucs 출력 폴더 탐색
                base = os.path.splitext(os.path.basename(wav_path))[0]
                sep_dir = os.path.join(tmp_demucs_dir, 'htdemucs', base)
                if os.path.exists(sep_dir):
                    for stem_name in os.listdir(sep_dir):
                        stem_path = os.path.join(sep_dir, stem_name)
                        if stem_path.endswith('.wav'):
                            key = stem_name.replace('.wav', '')
                            stems[key], _ = librosa.load(stem_path, sr=44100, mono=True)
                    demucs_ok = len(stems) > 0
                    print(f'[Demucs] 분리 완료: {list(stems.keys())}')
            else:
                print(f'[Demucs] 실패: {result.stderr[:200]}')
        except Exception as e:
            print(f'[Demucs] 에러: {e}')

        if not demucs_ok:
            # Demucs 실패 시 4-stem 대신 주파수 대역 분리로 폴백
            print('[폴백] 주파수 대역 분리 사용')
            # 저음역 (드럼/베이스)
            stems['drums'] = librosa.effects.preemphasis(y_full, coef=-0.97)
            # 나머지 (보컬/멜로디)
            stems['no_drums'] = librosa.effects.preemphasis(y_full, coef=0.97)

        # ── BPM 감지 ──
        bpm = 120
        tempo, beat_frames = librosa.beat.beat_track(y=y_full, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = tempo[0]
        bpm = int(round(float(tempo)))
        bpm = max(60, min(300, bpm))

        # ── 각 트랙별 onset 감지 ──
        def get_onsets(y_track, delta=0.1, min_gap=0.08):
            onset_env = librosa.onset.onset_strength(y=y_track, sr=sr)
            frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                                 backtrack=True, delta=delta)
            times = librosa.frames_to_time(frames, sr=sr)
            # 최소 간격 필터
            filtered = []
            for t in times:
                if len(filtered) == 0 or t - filtered[-1] >= min_gap:
                    filtered.append(t)
            return filtered

        # 드럼 트랙 → 레인 0, 1 (킥/스네어)
        drum_onsets = []
        if 'drums' in stems:
            drum_onsets = get_onsets(stems['drums'], delta=0.08, min_gap=0.1)
            print(f'[드럼] onset {len(drum_onsets)}개')

        # 보컬 트랙 → 레인 2
        vocal_onsets = []
        if 'vocals' in stems:
            vocal_onsets = get_onsets(stems['vocals'], delta=0.15, min_gap=0.15)
            print(f'[보컬] onset {len(vocal_onsets)}개')

        # 베이스 트랙 → 레인 0
        bass_onsets = []
        if 'bass' in stems:
            bass_onsets = get_onsets(stems['bass'], delta=0.12, min_gap=0.12)
            print(f'[베이스] onset {len(bass_onsets)}개')

        # 기타/피아노/멜로디 → 레인 3
        other_onsets = []
        if 'other' in stems:
            other_onsets = get_onsets(stems['other'], delta=0.1, min_gap=0.1)
            print(f'[기타/멜로디] onset {len(other_onsets)}개')

        # no_drums (폴백용)
        if 'no_drums' in stems and not vocal_onsets and not other_onsets:
            nd_onsets = get_onsets(stems['no_drums'], delta=0.1, min_gap=0.1)
            vocal_onsets = nd_onsets[:len(nd_onsets)//2]
            other_onsets = nd_onsets[len(nd_onsets)//2:]

        # ── 악기별 레인 매핑으로 노트 생성 ──
        notes = []
        used_times = set()

        def add_notes(onsets, lane, source):
            for t in onsets:
                t_r = round(float(t), 3)
                if t_r < 0.5 or t_r > duration - 0.5:
                    continue
                # 가까운 시간에 이미 노트가 있으면 스킵
                skip = False
                for ut in list(used_times):
                    if abs(ut - t_r) < 0.05:
                        skip = True
                        break
                if not skip:
                    used_times.add(t_r)
                    frame = min(librosa.time_to_frames(t_r, sr=sr, hop_length=512),
                               len(librosa.feature.rms(y=y_full)[0]) - 1)
                    rms_val = librosa.feature.rms(y=y_full)[0]
                    en = float(rms_val[frame] / (rms_val.max() or 1)) if frame < len(rms_val) else 0.5
                    notes.append({
                        't': t_r,
                        'l': lane,
                        'e': round(en, 2)
                    })

        # 드럼: 킥(강한 비트) → 레인 0, 스네어(약한 비트) → 레인 1
        for i, t in enumerate(drum_onsets):
            lane = 0 if i % 2 == 0 else 1  # 교대 배치
            t_r = round(float(t), 3)
            if t_r < 0.5 or t_r > duration - 0.5:
                continue
            used_times.add(t_r)
            notes.append({'t': t_r, 'l': lane, 'e': 0.8})

        # 베이스 → 레인 0 (드럼과 겹치지 않게)
        add_notes(bass_onsets, 0, 'bass')

        # 보컬 → 레인 2
        add_notes(vocal_onsets, 2, 'vocal')

        # 멜로디/기타 → 레인 3
        add_notes(other_onsets, 3, 'other')

        # 시간순 정렬
        notes.sort(key=lambda n: n['t'])

        # 같은 레인 연속 3회 이상 방지 (후처리)
        final_notes = []
        prev_lane = -1
        consecutive = 0
        for n in notes:
            if n['l'] == prev_lane:
                consecutive += 1
                if consecutive >= 3:
                    n['l'] = (n['l'] + 1 + consecutive) % 4
                    consecutive = 0
            else:
                consecutive = 0
            prev_lane = n['l']
            final_notes.append(n)

        # 정리
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        shutil.rmtree(tmp_demucs_dir, ignore_errors=True)

        method = 'Demucs' if demucs_ok else 'FreqBand'
        print(f'[분석 완료] BPM={bpm}, 노트={len(final_notes)}개, 길이={round(duration)}초 ({method})')
        return {
            'bpm': bpm,
            'duration': round(duration, 1),
            'notes': final_notes
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
