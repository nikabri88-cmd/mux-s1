# ==== MUX-S1 1.1, шаг 2b / ячейка 1: подготовка ================================================
# Шаг 2b прикрепляет признаки лиц к девяти сессиям FACE1500. Ни A, ни B он не считает.
# Ничего не устанавливает: признаки авторов уже переведены в простой файл внутри пакета кода,
# а ячейка 2 сверяет хэши исходных файлов авторов в Drive с теми, из которых делался перевод.
import os, sys, json, hashlib, zipfile, time
import numpy as np
from google.colab import drive
drive.mount("/content/drive")

ROOT = "/content/drive/MyDrive/mux_s1"
RAW = "/content/raw"; os.makedirs(RAW, exist_ok=True)
CODE_ZIP = f"{ROOT}/mux_s1_step2b_code.zip"
EXPECTED = {
 "mux_s1/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "mux_s1/core.py": "116c27ff8156773a06f8c7b7f3bb1820c423db59a2c7d13e5922d971f3b14b0f",
 "mux_s1/realdata.py": "87ca52f245c38856e075b6c0e8bdc1e6d791297f94dc0e27785e05fca688f419",
 "mux_s1/adapters.py": "137b6911f9f580186f963b35d37394c8ec95ffb26125440fbd1de831b392d2fe",
 "mux_s1/hesse_reader.py": "32e5655e9f6c9a1aedc2af9dbc664cd67e7a788abc760a6fe74014aa56ceb0c9",
 "mux_s1/faces.py": "2c2773ba27de3696668aab0b3c418957140ae79bfafd64815faed0a364bed44c",
 "mux_s1/feature_maps.npz": "816d6621771548e4968298ef7f627b62adba830ef02c6ce3448b95f92b0bb86d"
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
    print("Загрузи mux_s1_step2b_code.zip, который пришёл вместе с этим блокнотом:")
    up = files.upload()
    with open(CODE_ZIP, "wb") as f:
        f.write(next(iter(up.values())))
    assert code_ok(), "в загруженном zip не тот код – возьми zip, пришедший вместе с этим блокнотом"
zipfile.ZipFile(CODE_ZIP).extractall("/content/code2b")
if "/content/code2b" not in sys.path:
    sys.path.insert(0, "/content/code2b")
for m in [k for k in sys.modules if k == "mux_s1" or k.startswith("mux_s1.")]:
    del sys.modules[m]
from mux_s1 import hesse_reader as HR, adapters as AD, faces as FC
CACHE = f"{ROOT}/cache/reader_{EXPECTED['mux_s1/hesse_reader.py'][:12]}"   # тот же кэш, что в шаге 2
print("код: все хэши сходятся | кэш:", CACHE)
missing_cache = [n for n in HR.MANY_FACES if not os.path.exists(f"{CACHE}/{n}__primary.npz")]
assert not missing_cache, f"нет кэша шага 2 для {missing_cache}"

import requests
RID = "23604909"
meta = requests.get(f"https://zenodo.org/api/records/{RID}", timeout=60).json()
fl = meta.get("files") or []
fl = fl.get("entries", fl) if isinstance(fl, dict) else fl
fl = list(fl.values()) if isinstance(fl, dict) else fl
by_name = {(f.get("key") or f.get("filename")): f for f in fl}


def download(fname, dest):
    f = by_name[fname]
    links = f.get("links") or {}
    url = links.get("content") or links.get("self") or f"https://zenodo.org/records/{RID}/files/{fname}?download=1"
    md5_want = (f.get("checksum") or "").replace("md5:", "")
    for attempt in range(3):
        try:
            h = hashlib.md5()
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as out:
                    for chunk in r.iter_content(1 << 24):
                        out.write(chunk); h.update(chunk)
            if md5_want and h.hexdigest() != md5_want:
                raise IOError("md5 не сходится")
            return
        except Exception as e:
            print(f"   попытка {attempt + 1}: {e}")
    raise IOError(f"{fname}: не скачался")


# ==== ячейка 2: признаки лиц. Скопируй вывод и пришли =========================================
STIM = f"{ROOT}/data/stimuli"
for fs, want in FC.source_hashes().items():
    got = hashlib.sha256(open(f"{STIM}/{fs}/{FC.MAP_FILE}", "rb").read()).hexdigest()
    print(f"исходник признаков {fs}: {'совпадает с переведённым' if got == want else 'НЕ СОВПАДАЕТ'}")
    assert got == want
maps, rows, gaps = {}, [], {}
for name in HR.MANY_FACES:
    out = f"{CACHE}/{name}__faces.npz"
    s = AD.load_session(f"{CACHE}/{name}__primary.npz")
    plist = s.meta["priming_list"]
    if not os.path.exists(out):
        t0 = time.time()
        mat = f"{RAW}/{name}.mat"
        print(f"{name}: качаю ради имени XML...", flush=True)
        download(name + ".mat", mat)
        xml = FC.read_xml(mat, name)
        os.remove(mat)
        fs = FC.set_for_xml(xml)
        if fs not in maps:
            maps[fs] = FC.load_feature_npz(fs)
        F, missing = FC.faces_matrix(plist, maps[fs])
        FC.save_faces(out, F, fs, xml, missing, plist)
        print(f"   готово за {time.time() - t0:.0f} с", flush=True)
    z = np.load(out)
    F, fm = z["faces"], json.loads(str(z["__meta__"]))
    used = FC.used_faces(s)
    gap = [plist[i] for i in used if not np.all(np.isfinite(F[i]))]
    gaps[name] = gap
    rows.append([name.replace("session_", ""), fm["xml_filename"].split("\\")[-1], fm["feature_set"],
                 len(plist), len(used), len(gap), F.shape[1]])

head = ["сессия", "XML", "набор признаков", "стимулов в прайминге", "лиц в парах", "из них без признаков", "признаков"]
print("\n| " + " | ".join(head) + " |"); print("|" + "---|" * len(head))
for r in rows:
    print("| " + " | ".join(str(x) for x in r) + " |")
for n, g in gaps.items():
    if g:
        print(f"\n{n}: лица в парах без признаков (первые 5): {g[:5]}")
ok = all(not g for g in gaps.values())
print("\nИтог:", "у всех лиц из пар есть признаки, блок усиления может идти" if ok
      else "есть лица без признаков – пришли вывод, правило решим до шага 3")
