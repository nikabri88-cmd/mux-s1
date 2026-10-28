# ==== MUX-S1 1.1, шаг 4 (поисковый E-repeat) / ячейка 1: подготовка ================================================
# Поисковый прогон после основного расчёта; вердиктов не выносит. Ничего не устанавливает.
import os, sys, json, hashlib, zipfile, subprocess, time
from google.colab import drive
drive.mount("/content/drive")

ROOT = "/content/drive/MyDrive/mux_s1"
CODE_ZIP = f"{ROOT}/mux_s1_step4_code.zip"
CODE = "/content/code4"
EXPECTED = {
 "mux_s1/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "mux_s1/core.py": "116c27ff8156773a06f8c7b7f3bb1820c423db59a2c7d13e5922d971f3b14b0f",
 "mux_s1/realdata.py": "87ca52f245c38856e075b6c0e8bdc1e6d791297f94dc0e27785e05fca688f419",
 "mux_s1/adapters.py": "100c6e67f1dec7abafd8cd58eea0e6e31f6094ec2b3ea5ef253c9b617396edaa",
 "mux_s1/hesse_reader.py": "32e5655e9f6c9a1aedc2af9dbc664cd67e7a788abc760a6fe74014aa56ceb0c9",
 "mux_s1/faces.py": "2c2773ba27de3696668aab0b3c418957140ae79bfafd64815faed0a364bed44c",
 "verify_real.py": "dadcbd2327a23e2a93a8804e7548d4ac9a7f02a91aab94c3629b0a247bcc949b",
 "explore_repeat.py": "9d63f9088ae1890b9244f91d96f3b92637d721a8c285b09713b33f041c23154c"
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
    print("Загрузи mux_s1_step4_code.zip, который пришёл вместе с этим блокнотом:")
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


# ==== ячейка 2: поисковый прогон E-repeat и комментарии об инъекциях ===========================
t0 = time.time()
r = subprocess.run([sys.executable, "explore_repeat.py", "--cache", CACHE, "--out", f"{RES}/explore"],
                   cwd=CODE, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(r.stdout[-8000:])
open(f"{RES}/explore_log.txt", "w").write(r.stdout)
print(f"E-repeat: {'готово' if r.returncode == 0 else f'ОШИБКА, код {r.returncode}'} за {(time.time() - t0) / 60:.0f} мин")
print("Напиши мне «готово» – результаты я заберу из Drive сам.")
