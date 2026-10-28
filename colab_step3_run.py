# ==== MUX-S1 1.1, шаг 3 / ячейка 1: подготовка ================================================
# Запускать после поста в закрытом канале. Ничего не устанавливает.
import os, sys, json, hashlib, zipfile, subprocess, time
from google.colab import drive
drive.mount("/content/drive")

ROOT = "/content/drive/MyDrive/mux_s1"
CODE_ZIP = f"{ROOT}/mux_s1_step3_code.zip"
CODE = "/content/code3"
EXPECTED = {
 "mux_s1/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "mux_s1/core.py": "116c27ff8156773a06f8c7b7f3bb1820c423db59a2c7d13e5922d971f3b14b0f",
 "mux_s1/realdata.py": "87ca52f245c38856e075b6c0e8bdc1e6d791297f94dc0e27785e05fca688f419",
 "mux_s1/adapters.py": "100c6e67f1dec7abafd8cd58eea0e6e31f6094ec2b3ea5ef253c9b617396edaa",
 "mux_s1/hesse_reader.py": "32e5655e9f6c9a1aedc2af9dbc664cd67e7a788abc760a6fe74014aa56ceb0c9",
 "mux_s1/faces.py": "2c2773ba27de3696668aab0b3c418957140ae79bfafd64815faed0a364bed44c",
 "verify_real.py": "dadcbd2327a23e2a93a8804e7548d4ac9a7f02a91aab94c3629b0a247bcc949b",
 "verify_secondary.py": "2663221353f5992814e1a2ced40cf30df59276522e89f220b156ab4824bc5bcf"
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
    print("Загрузи mux_s1_step3_code.zip, который пришёл вместе с этим блокнотом:")
    up = files.upload()
    with open(CODE_ZIP, "wb") as f:
        f.write(next(iter(up.values())))
    assert code_ok(), "в загруженном zip не тот код – возьми zip, пришедший вместе с этим блокнотом"
zipfile.ZipFile(CODE_ZIP).extractall(CODE)
SHA = EXPECTED["mux_s1/hesse_reader.py"][:12]
CACHE = f"{ROOT}/cache/reader_{SHA}"
RES = f"{ROOT}/results/reader_{SHA}"
os.makedirs(RES, exist_ok=True)
print("код: все хэши сходятся")
print("замороженные файлы:", EXPECTED["verify_real.py"][:16], "verify_real.py |",
      EXPECTED["mux_s1/realdata.py"][:16], "realdata.py |", EXPECTED["mux_s1/core.py"][:16], "core.py")


# ==== ячейка 2: проверка, что всё на месте (ничего не считает) ================================
sys.path.insert(0, CODE)
for m in [k for k in sys.modules if k == "mux_s1" or k.startswith("mux_s1.")]:
    del sys.modules[m]
from mux_s1 import hesse_reader as HR, adapters as AD, faces as FC
need = [f"{n}__{v}.npz" for n in HR.REC_IT for v in HR.VARIANTS]
miss = [f for f in need if not os.path.exists(f"{CACHE}/{f}")]
print("кэш шага 2:", "все", len(need), "файлов на месте" if not miss else f"НЕ ХВАТАЕТ {miss[:5]}")
fmiss = [n for n in HR.MANY_FACES if not os.path.exists(f"{CACHE}/{n}__faces.npz")]
print("признаки шага 2b:", "все 9 на месте" if not fmiss else f"НЕТ для {fmiss} – сначала шаг 2b")
assert not miss and not fmiss


# ==== ячейка 3: основной расчёт – замороженный verify_real.py, команда из пререгистрации ========
t0 = time.time()
r = subprocess.run([sys.executable, "verify_real.py", "--source", "dandi", "--path", CACHE,
                    "--design", "FACE1500", "--out", f"{RES}/primary"], cwd=CODE, text=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(r.stdout[-6000:])
print(f"основной расчёт: {'готово' if r.returncode == 0 else 'ОШИБКА'} за {(time.time() - t0) / 60:.0f} мин")
open(f"{RES}/primary_log.txt", "w").write(r.stdout)
assert r.returncode == 0


# ==== ячейка 4: вторичные расчёты (объявлены в журнале, вердиктов не выносят) ===================
t0 = time.time()
r = subprocess.run([sys.executable, "verify_secondary.py", "--cache", CACHE, "--out", f"{RES}/secondary",
                    "--primary", f"{RES}/primary/verify.json"], cwd=CODE, text=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(r.stdout[-4000:])
print(f"вторичные: {'готово' if r.returncode == 0 else 'ОШИБКА'} за {(time.time() - t0) / 60:.0f} мин")
open(f"{RES}/secondary_log.txt", "w").write(r.stdout)


# ==== ячейка 5: архив результатов с хэшем ===================================================
import shutil
arc = shutil.make_archive(f"{ROOT}/mux_s1_v11_results_{SHA}", "zip", RES)
h = hashlib.sha256(open(arc, "rb").read()).hexdigest()
open(f"{RES}/ARCHIVE.sha256", "w").write(f"{h}  {os.path.basename(arc)}\n")
print("архив результатов:", os.path.basename(arc))
print("sha256:", h)
print("\nГотово. Напиши мне «готово» – результаты я заберу из Drive сам.")
