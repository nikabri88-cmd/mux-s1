# MUX-S1 1.1, шаг 1: только структура одного файла выпуска. Значения не печатаются.
#
# Что делает ячейка:
#   1. берёт список файлов записи Zenodo (имена и размеры);
#   2. скачивает самый маленький session_*.mat;
#   3. печатает дерево файла: пути, формы, типы, класс MATLAB. Ни одного числа из данных.
#   4. печатает описание записи и README, если он есть: это документация, как раздел методов.
#
# Единственное, что надо вписать: ссылку на запись Zenodo в строке ниже.

ZENODO = "https://zenodo.org/records/23604909"

import os, re, json, requests, numpy as np

rid = re.findall(r"\d{5,}", ZENODO)
assert rid, "Вставь в ZENODO ссылку на запись Zenodo, например https://zenodo.org/records/1234567"
rid = rid[-1]
meta = requests.get(f"https://zenodo.org/api/records/{rid}", timeout=60).json()
files = meta.get("files") or meta.get("entries") or []
files = files.get("entries", files) if isinstance(files, dict) else files
files = list(files.values()) if isinstance(files, dict) else files


def fname(f): return f.get("key") or f.get("filename")
def fsize(f): return int(f.get("size") or f.get("filesize") or 0)
def furl(f):
    links = f.get("links") or {}
    return links.get("content") or links.get("self") or f"https://zenodo.org/records/{rid}/files/{fname(f)}?download=1"


print(f"Запись {rid}: {meta.get('metadata', {}).get('title', '')}")
print(f"\nФайлы записи ({len(files)}):")
for f in sorted(files, key=fname):
    print(f"  {fname(f):<48} {fsize(f) / 1e9:7.3f} ГБ")

desc = re.sub(r"<[^>]+>", " ", meta.get("metadata", {}).get("description", "") or "")
print("\nОписание записи (документация):\n", re.sub(r"\s+", " ", desc).strip()[:4000])

readmes = [f for f in files if re.match(r"readme", fname(f), re.I)]
for f in readmes[:2]:
    if fsize(f) < 2e6:
        print(f"\n===== {fname(f)} =====\n", requests.get(furl(f), timeout=120).text[:8000])

sessions = [f for f in files if re.match(r"session.*\.mat$", fname(f), re.I)]
assert sessions, "В записи не нашлось session_*.mat – пришли мне список файлов выше."
pick = min(sessions, key=fsize)
path = os.path.join("/content", fname(pick))
if not os.path.exists(path) or os.path.getsize(path) != fsize(pick):
    print(f"\nСкачиваю {fname(pick)} ({fsize(pick) / 1e9:.2f} ГБ)...")
    with requests.get(furl(pick), stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as out:
            for chunk in r.iter_content(1 << 24):
                out.write(chunk)


# ---------------------------------------------------------------- печать структуры, без значений
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


def structure_v5(p, max_lines=400):
    from scipy.io import loadmat, whosmat
    lines = [f"  {n:<56} {str(s):<18} {c}" for n, s, c in whosmat(p)]
    m = loadmat(p, squeeze_me=False, struct_as_record=False)

    def walk(node, name, depth):
        if len(lines) >= max_lines or depth > 5:
            return
        a = np.asarray(node)
        if a.dtype == object and a.size and hasattr(a.flat[0], "_fieldnames"):
            for k in a.flat[0]._fieldnames:
                walk(getattr(a.flat[0], k), f"{name}/{k}", depth + 1)
        elif a.dtype == object and a.size:
            first = np.asarray(a.flat[0])
            lines.append(f"  {name:<56} {str(a.shape):<18} ячейка, первый элемент {first.shape} {first.dtype}")
        elif depth > 1:
            lines.append(f"  {name:<56} {str(a.shape):<18} {a.dtype}")

    for k, v in m.items():
        if not k.startswith("__"):
            walk(v, k, 1)
    return lines


v73 = is_v73(path)
print(f"\n===== {fname(pick)}: {'MATLAB 7.3 / HDF5' if v73 else 'MATLAB 5'}, "
      f"{os.path.getsize(path) / 1e9:.2f} ГБ =====")
print(f"  {'путь':<64} {'форма (MATLAB)':<20} тип / класс / текст")
print("\n".join(structure_v73(path) if v73 else structure_v5(path)))
print("\nГотово. Скопируй всё, что выше, и пришли мне. Числовых значений из данных в выводе нет:"
      "\nу чисел только формы и типы, текст показан только у текстовых полей (названия, имена файлов, комментарии).")
