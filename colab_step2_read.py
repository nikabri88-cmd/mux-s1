# ==== MUX-S1 1.1, шаг 2 / ячейка 1: подготовка =================================================
# Шаг 2 читает все 19 сессий и сохраняет счётчики спайков в Google Drive.
# Ни одной проверки A и ни одного предсказания B этот шаг НЕ считает: только сантехнику
# (сколько проб, нейронов, пар, какая частота разряда) – величины, которых нет в предсказаниях.

import os, sys, json, hashlib, zipfile, time, re
from google.colab import drive
drive.mount("/content/drive")

ROOT = "/content/drive/MyDrive/mux_s1"            # всё сохраняется сюда
CACHE = f"{ROOT}/cache"                            # счётчики по сессиям (.npz), ~ сотни МБ
RAW = "/content/raw"                               # сюда качается один .mat, потом удаляется
CODE_ZIP = f"{ROOT}/mux_s1_step2_code.zip"
os.makedirs(CACHE, exist_ok=True); os.makedirs(RAW, exist_ok=True)

EXPECTED = {
 "mux_s1/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "mux_s1/core.py": "116c27ff8156773a06f8c7b7f3bb1820c423db59a2c7d13e5922d971f3b14b0f",
 "mux_s1/realdata.py": "87ca52f245c38856e075b6c0e8bdc1e6d791297f94dc0e27785e05fca688f419",
 "mux_s1/adapters.py": "d49b8a173ef19799a63cbbc4ca9ed2b174c60a3df9c1540059c1084b8d1ba5bd",
 "mux_s1/hesse_reader.py": "32e5655e9f6c9a1aedc2af9dbc664cd67e7a788abc760a6fe74014aa56ceb0c9"
}


def code_ok():
    if not os.path.exists(CODE_ZIP):
        return False
    z = zipfile.ZipFile(CODE_ZIP)
    try:
        return all(hashlib.sha256(z.read(rel)).hexdigest() == want for rel, want in EXPECTED.items())
    except KeyError:
        return False


if not code_ok():
    from google.colab import files
    print("Загрузи mux_s1_step2_code.zip, который пришёл вместе с этим блокнотом:")
    up = files.upload()
    os.makedirs(ROOT, exist_ok=True)
    with open(CODE_ZIP, "wb") as f:
        f.write(next(iter(up.values())))
    assert code_ok(), "в загруженном zip не тот код – возьми zip, пришедший вместе с этим блокнотом"

zipfile.ZipFile(CODE_ZIP).extractall("/content/code")
if "/content/code" not in sys.path:
    sys.path.insert(0, "/content/code")
for m in [k for k in sys.modules if k == "mux_s1" or k.startswith("mux_s1.")]:
    del sys.modules[m]                             # при перезапуске не брать старую версию из памяти
print("код: все хэши сходятся")

from mux_s1 import hesse_reader as HR, adapters as AD
READER_SHA = EXPECTED["mux_s1/hesse_reader.py"][:12]
CACHE = f"{CACHE}/reader_{READER_SHA}"            # кэш привязан к версии читателя
os.makedirs(CACHE, exist_ok=True)
print("сессий в списке IT авторов:", len(HR.REC_IT), "| варианты:", list(HR.VARIANTS))
print("кэш этой версии читателя:", CACHE)


# ==== ячейка 2: чтение сессий (можно перезапускать, сделанное пропускается) ====================
import requests
RID = "23604909"
meta = requests.get(f"https://zenodo.org/api/records/{RID}", timeout=60).json()
files_meta = meta.get("files") or []
files_meta = files_meta.get("entries", files_meta) if isinstance(files_meta, dict) else files_meta
files_meta = list(files_meta.values()) if isinstance(files_meta, dict) else files_meta
by_name = {(f.get("key") or f.get("filename")): f for f in files_meta}


def url_of(f):
    links = f.get("links") or {}
    return links.get("content") or links.get("self") or f"https://zenodo.org/records/{RID}/files/{f['key']}?download=1"


def download(fname, dest):
    f = by_name[fname]
    size = int(f.get("size") or 0)
    md5_want = (f.get("checksum") or "").replace("md5:", "")
    for attempt in range(3):
        try:
            h = hashlib.md5()
            with requests.get(url_of(f), stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as out:
                    for chunk in r.iter_content(1 << 24):
                        out.write(chunk); h.update(chunk)
            if size and os.path.getsize(dest) != size:
                raise IOError("размер не сходится")
            if md5_want and h.hexdigest() != md5_want:
                raise IOError("md5 не сходится")
            return
        except Exception as e:
            print(f"   попытка {attempt + 1}: {e}")
    raise IOError(f"{fname}: не скачался")


t_all = time.time()
for name in HR.REC_IT:
    targets = {v: f"{CACHE}/{name}__{v}.npz" for v in HR.VARIANTS}
    if all(os.path.exists(p) for p in targets.values()):
        print(f"{name}: уже в кэше"); continue
    t0 = time.time()
    mat = f"{RAW}/{name}.mat"
    print(f"{name}: качаю {int(by_name[name + '.mat']['size']) / 1e9:.2f} ГБ...", flush=True)
    download(name + ".mat", mat)
    print(f"   скачан за {time.time() - t0:.0f} с, читаю...", flush=True)
    sessions = HR.read_variants(mat, name=name)
    for v, s in sessions.items():
        AD.save_session(s, targets[v])
    os.remove(mat)
    m = sessions["primary"].meta
    print(f"   готово за {time.time() - t0:.0f} с: нейронов {m['n_units_used']}, пар {m['n_pairs_used']}"
          f" из {m['n_pairs_all']}, проб в файле {m['n_trials_file']}", flush=True)
print(f"\nВсе сессии в кэше. Общее время {(time.time() - t_all) / 60:.0f} мин.")


# ==== ячейка 3: сантехника. Скопируй вывод этой ячейки целиком и пришли ======================
rows, comments = [], {}
for name in HR.REC_IT:
    s = AD.load_session(f"{CACHE}/{name}__primary.npz")
    m = s.meta
    pr = AD.plumbing_report([s])[0]
    grp = "FACE1500" if name in HR.MANY_FACES else "comb"
    rows.append([name.replace("session_", ""), grp, m["rec_it"]["paradigm"],
                 "+".join(map(str, m["rec_it"]["probes"])),
                 m["n_units_used"], m["n_units_session"], m["n_trials_file"], m["n_priming"],
                 f"{m['n_binocular_valid']}/{m['n_binocular_all']}", m["n_unambiguous_valid"],
                 f"{m['n_pairs_used']}/{m['n_pairs_all']}", m["n_thrown_away"],
                 "да" if m["many_stimuli_exception"] else "", pr["mean_rate_hz"],
                 m["raster_source"].split("_")[0], len(m["injection_comments"])])
    comments[name] = m["comments"]
head = ["сессия", "группа", "пар-ма", "зонды", "нейр.исп", "нейр.всего", "проб", "прайм",
        "бино годн/всего", "однозн", "пары исп/всего", "выброш", "искл.fsm", "Гц", "растр", "инъекц"]
print("| " + " | ".join(head) + " |")
print("|" + "---|" * len(head))
for r in rows:
    print("| " + " | ".join(str(x) for x in r) + " |")
for grp in ("FACE1500", "comb"):
    sel = [r for r in rows if r[1] == grp]
    print(f"\n{grp}: сессий {len(sel)}, нейронов {sum(r[4] for r in sel)}, "
          f"пар {sum(int(r[10].split('/')[0]) for r in sel)}")
print("\nКомментарии в файлах (все сессии):")
for name, cs in comments.items():
    for c in cs:
        print(f"  {name.replace('session_', '')}: {c}")
skipped = {n: AD.load_session(f"{CACHE}/{n}__primary.npz").meta["skipped"] for n in HR.REC_IT}
print("\nПары, выпавшие из-за пустой ячейки:", {k.replace("session_", ""): len(v) for k, v in skipped.items() if v})


# ==== ячейка 4: признаки лиц из data.zip. Скопируй вывод и пришли ===============================
# Извлекается ТОЛЬКО папка data/stimuli (описания стимулов, по которым модель авторов рисует лица).
# Сводки и результаты авторов (data/python_summaries, data/results) не извлекаются и не открываются;
# у них печатаются только имена файлов и размеры.
import numpy as np
ZIP = "/content/data.zip"
if not os.path.exists(f"{ROOT}/stimuli_extracted.flag"):
    if not os.path.exists(ZIP):
        print("качаю data.zip (2.1 ГБ)..."); download("data.zip", ZIP)
    z = zipfile.ZipFile(ZIP)
    tops = {}
    for i in z.infolist():
        parts = i.filename.split("/")
        key = "/".join(parts[:3]) if len(parts) > 3 else i.filename
        tops.setdefault(key, [0, 0]); tops[key][0] += 1; tops[key][1] += i.file_size
    print("Содержимое data.zip (папки и файлы, только имена и размеры):")
    for k in sorted(tops):
        print(f"  {k:<70} файлов {tops[k][0]:>5}  {tops[k][1] / 1e6:9.1f} МБ")
    stim = [i for i in z.infolist() if i.filename.startswith("data/stimuli/") and not i.is_dir()]
    for i in stim:
        z.extract(i, ROOT)
    open(f"{ROOT}/stimuli_extracted.flag", "w").write(str(len(stim)))
    os.remove(ZIP)
    print(f"\nИзвлечено файлов из data/stimuli: {len(stim)}")

def is_v73(p):
    with open(p, "rb") as fh:
        return b"MATLAB 7.3" in fh.read(128)


TEXT_MAX = 160        # символов текста на одну подпись
CELL_TEXT_SHOW = 6    # сколько строк показать из длинной ячейки строк
CELL_ITEMS_SHOW = 40  # сколько элементов ячейки структур разобрать


def structure_v73(p, max_lines=2500, max_depth=8):
    """Дерево файла MATLAB 7.3 со спуском в ячейки и структуры.

    Числовые поля: только форма (как в MATLAB), тип и класс. Текстовые поля (класс char):
    сам текст, обрезанный до TEXT_MAX, – это названия парадигм, имена файлов стимулов и
    комментарии, без них разметку проб не восстановить. Измерений среди них нет.
    """
    import h5py
    lines = []
    with h5py.File(p, "r") as h:
        def cls(node):
            v = node.attrs.get("MATLAB_class", b"")
            return v.decode() if isinstance(v, bytes) else str(v)

        def text(ds):
            a = np.asarray(ds[()]).reshape(-1)
            return "".join(chr(int(c)) for c in a[:TEXT_MAX])

        def put(s):
            if len(lines) < max_lines:
                lines.append(s)

        def walk(node, name, depth):
            if len(lines) >= max_lines or depth > max_depth:
                return
            if isinstance(node, h5py.Group):
                put(f"  {name + '/':<64} {'структура':<20} {cls(node)}")
                for k in node.keys():
                    walk(node[k], f"{name}/{k}", depth + 1)
                return
            c = cls(node)
            shape = tuple(reversed(node.shape))                      # как видит MATLAB
            if "MATLAB_empty" in node.attrs:
                put(f"  {name:<64} {'пусто':<20} {c}")
            elif c == "char":
                put(f"  {name:<64} {'текст':<20} «{text(node)}»")
            elif node.dtype == h5py.ref_dtype:                       # ячейка
                refs = np.asarray(node[()]).T.reshape(-1, order="F")  # порядок элементов MATLAB
                put(f"  {name:<64} {str(shape):<20} cell, элементов {refs.size}")
                if refs.size == 0:
                    return
                first = h[refs[0]]
                if isinstance(first, h5py.Dataset) and cls(first) == "char":
                    for i, r in enumerate(refs[:CELL_TEXT_SHOW]):
                        put(f"  {name + '{' + str(i + 1) + '}':<64} {'текст':<20} «{text(h[r])}»")
                    if refs.size > CELL_TEXT_SHOW:
                        put(f"  {'':<64} {'':<20} … ещё {refs.size - CELL_TEXT_SHOW} строк")
                else:
                    for i, r in enumerate(refs[:CELL_ITEMS_SHOW]):
                        walk(h[r], f"{name}{{{i + 1}}}", depth + 1)
                    if refs.size > CELL_ITEMS_SHOW:
                        put(f"  {'':<64} {'':<20} … ещё {refs.size - CELL_ITEMS_SHOW} элементов")
            else:
                put(f"  {name:<64} {str(shape):<20} {str(node.dtype):<8} {c}")

        for k in h.keys():
            if not k.startswith("#"):
                walk(h[k], k, 1)
    if len(lines) >= max_lines:
        lines.append(f"  … вывод обрезан на {max_lines} строках")
    return lines




STIM = f"{ROOT}/data/stimuli"
print("\nФайлы признаков:")
for dp, _, fs in os.walk(STIM):
    for fn in sorted(fs):
        p = os.path.join(dp, fn)
        print(f"  {os.path.relpath(p, STIM):<70} {os.path.getsize(p) / 1e6:8.2f} МБ")
fm_files = sorted(os.path.join(dp, fn) for dp, _, fs in os.walk(STIM) for fn in fs
                  if fn.startswith("feat_map_priming_reconstructed_0.00") and fn.endswith(".mat"))
for p in fm_files:
    print(f"\n===== {os.path.relpath(p, STIM)}: {'MATLAB 7.3' if is_v73(p) else 'MATLAB 5/7'} =====")
    if is_v73(p):
        print("\n".join(structure_v73(p, max_lines=200)))
    else:
        from scipy.io import whosmat
        for n, shape, cls in whosmat(p):
            print(f"  переменная {n}: форма {shape}, класс {cls}")

# containers.Map читается пакетом mat-io; печатаются только число ключей, три имени стимулов
# и длина вектора признаков – без самих значений
!pip -q install mat-io
try:
    from matio import load_from_mat
    for p in fm_files:
        d = load_from_mat(p)
        fm = d["feat_map"] if isinstance(d, dict) else getattr(d, "feat_map")
        keys = list(fm.keys()) if hasattr(fm, "keys") else None
        print(f"\n{os.path.relpath(p, STIM)}: тип {type(fm).__name__}",
              f"ключей {len(keys)}" if keys else "")
        if keys:
            v = np.asarray(fm[keys[0]])
            print("  первые имена:", keys[:3])
            print("  вектор признаков: форма", v.shape, "тип", v.dtype)
except Exception as e:
    print("\nmat-io не прочитал containers.Map:", type(e).__name__, str(e)[:300])
