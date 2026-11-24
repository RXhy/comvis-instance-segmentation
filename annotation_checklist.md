# Checklist anotasi mycapture (61 foto)

Daftar ini dibuat dari melihat thumbnail setiap foto, jadi **perkiraan**: cek lagi saat anotasi.
Nama kelas persis seperti di `outputs/class_names_v5.txt`. "×n" = perkiraan jumlah instance.

## Aturan keputusan (pakai konsisten, catat di logbook)

| Benda di foto | Kelas yang dipakai |
|---|---|
| Dinding cat/plester (termasuk panel ballroom) | `wall-other-merged` |
| Dinding bata / keramik (termasuk backsplash mozaik dapur) / kayu | `wall-brick` / `wall-tile` / `wall-wood` |
| Lantai keramik, beton | `floor-other-merged` |
| Lantai kayu / motif kayu | `floor-wood` |
| Karpet, keset, karpet ballroom | `rug-merged` |
| Plafon | `ceiling-merged` |
| Kursi apa pun, **termasuk bangku/stool tanpa sandaran** dan kursi kuliah | `chair` |
| Bangku kayu panjang berbantal (#42, #43), sofa | `couch` |
| Meja bundar makan (#24, #25) | `dining table` |
| Meja lain (meja belajar, meja kerja, meja kafe, meja samping) | `table-merged` |
| Lemari, lemari laci (kayu/plastik), loker, kabinet dapur | `cabinet-merged` |
| Rak terbuka (besi/kayu, tanpa pintu) | `shelf` |
| Kardus, kotak sepatu, kontainer plastik bertutup, kotak styrofoam | `box` |
| Botol, toples, botol semprot, tumbler, galon | `bottle` |
| Gelas, mug | `cup` |
| Monitor komputer, TV | `tv` |
| HP, tablet | `cell phone` |
| Tas tangan, tas belanja/tote | `handbag` |
| Ransel | `backpack` |
| Lampu meja, lampu gantung, downlight plafon, lampu sorot | `light` |
| Stopkontak / saklar | `plug` |
| Jam dinding, jam digital | `clock` |
| Tanaman di pot | `potted plant` |
| Tanaman di botol/vas kaca berisi air (#41, #49) | `vase` |
| Figur/mainan | `toy` |
| Kertas lepas, sticky notes, voucher | `paper` |

**Tidak ada kelasnya, biarkan tanpa label:** kipas angin, mesin cuci, AC, speaker, rice cooker, akuarium,
sangkar burung, helm, kantong plastik/kresek, karung, ember, keranjang cucian, gantungan/jemuran baju,
pakaian, poster/lukisan/kaligrafi, kabel, sapu/pengki, tangga lipat, selang, tudung saji, kendaraan di luar jendela,
pohon/langit di luar, CCTV.

**Kelas yang muncul di fotomu tapi tidak punya contoh di train v5:** `keyboard`, `mouse` (#51).
Tetap dianotasi; model pasti gagal di kelas ini, dan itu bahan analisis.

## Per foto

| # | Anotasi (kelas ×perkiraan jumlah) | Biarkan (tanpa kelas) | Catatan |
|---|---|---|---|
| 01 | wall-other-merged, table-merged, bottle ×4, box ×2, paper ×1, toy ×2 | kabel, kain | voucher: tutup/blur kodenya |
| 02 | wall-other-merged, cabinet-merged, box ×7, floor-wood | kantong plastik, tas gitar, sapu, poster | |
| 03 | wall-other-merged, cabinet-merged (meja laci rendah), light (lampu meja), box ×4, window-other, door-stuff, ceiling-merged, floor-wood | kabel, adaptor | |
| 04 | wall-other-merged, window-other, bed, pillow ×3, blanket, rug-merged | | sudah dikerjakan |
| 05 | wall-other-merged, box ×2 (kardus Fender, kontainer hijau), cabinet-merged (laci plastik), bed (kasur), floor-other-merged | kipas, rak plastik | |
| 06 | wall-other-merged, window-other, pillow, bed, plug, light, door-stuff, floor-other-merged | | backlight |
| 07 | wall-other-merged, refrigerator, box ×6, handbag ×1 | kantong plastik, peti kayu, sangkar | sangat penuh |
| 08 | wall-other-merged, box ×6, handbag ×1 | sangkar burung, peti kayu, plastik | mirip #07 |
| 09 | wall-other-merged, window-other ×2, box ×4, bottle (semprot), floor-wood | karung, tangga, papan, filter air | backlight |
| 10 | potted plant ×5, wall-brick | pagar, atap, langit, pohon luar | |
| 11 | wall-other-merged, window-other, box ×3, bottle, floor-wood | karung, tangga, papan | mirip #09 |
| 12 | box ×6, floor-wood, rug-merged | rice cooker, karung, plastik | sudut atas |
| 13 | wall-other-merged, box ×6, handbag/backpack ×1 | plastik, kawat, peti | mirip #07 |
| 14 | wall-other-merged, floor-other-merged, table-merged, chair (stool), handbag ×2, box (styrofoam), window-other | ember, pengki, sapu, jemuran | |
| 15 | wall-other-merged, floor-other-merged, bottle ×5, box ×3, door-stuff | mesin cuci, keranjang, jemuran, karung, ember cat | |
| 16 | wall-tile, sink, toilet, bottle | rak sabun, selang | sudah dibahas |
| 17 | wall-tile, sink, toilet, bottle | rak sabun, selang | mirip #16 |
| 18 | wall-other-merged, box ×2, door-stuff, floor-other-merged, potted plant (luar pintu), curtain (kain hitam kiri) | karung, ember cat, selang, sapu | |
| 19 | wall-other-merged, wall-brick, stove-merged, box ×3, shelf, plate (tumpukan), handbag | akuarium, selang gas | |
| 20 | shelf ×2, box ×20+, bottle ×10+, toy (mainan anak), handbag ×2 | AC, plastik, karung | **sangat berat**, pertimbangkan diganti |
| 21 | wall-other-merged, ceiling-merged, shelf, box ×10+, bottle | sangkar, AC, plastik | berat |
| 22 | ceiling-merged, box ×6 (termasuk styrofoam), shelf, handbag | sangkar, plastik | berat |
| 23 | wall-other-merged, ceiling-merged, shelf, box ×10+, handbag, bottle | sangkar, AC | mirip #21–22 |
| 24 | chair, dining table, floor-other-merged, wall-brick, bottle ×6 (toples, galon), cup ×2 | tudung saji | |
| 25 | dining table, chair ×3, backpack/handbag (tas hitam), bottle ×10, cup ×3, wall-brick, floor-other-merged, food-other-merged | tudung saji, keranjang | penuh di atas meja |
| 26 | cabinet-merged ×3, wall-brick, wall-tile, sink, counter, stove-merged, bottle ×5, plate ×10+ (rak piring), cup, spoon, pan | rak bumbu, talenan | penuh |
| 27 | wall-brick, cabinet-merged, wall-tile, counter, stove-merged, pan ×2, bottle, plate | plastik gantung, gayung | |
| 28 | cabinet-merged ×3, wall-tile, sink, counter, plate ×10+, bottle ×5, cup, spoon | rak bumbu | mirip #26 |
| 29 | wall-other-merged, backpack, bottle, box (krat biru, opsional) | helm, bola, keranjang anyaman, lukisan, plastik | |
| 30 | wall-other-merged, chair, tv, table-merged, shelf, box ×10 (kotak sepatu), toy (di etalase), bottle, floor-wood, door-stuff | etalase akrilik, kain | |
| 31 | wall-other-merged, bed, pillow ×3 | pakaian, keranjang cucian | |
| 32 | cabinet-merged, box ×4, handbag ×3, wall-tile (panel hitam) | plastik | |
| 33 | wall-other-merged, cabinet-merged, box (kotak food processor) | poster, wadah beras, kain | |
| 34 | wall-other-merged, shelf, book ×6, clock (jam digital), floor-other-merged | poster, sajadah | |
| 35 | wall-other-merged, cabinet-merged, box, clock | kaligrafi, poster, wadah beras | |
| 36 | wall-other-merged, cabinet-merged, shelf, book, box | kaligrafi, poster, pakaian | |
| 37 | wall-other-merged, box ×3, handbag | frisbee, gulungan kain | sebagian besar dinding kosong |
| 38 | wall-brick, window-other, potted plant ×3, bottle | akuarium, lukisan, meja mesin jahit | |
| 39 | wall-brick, window-other | ukiran kayu, hiasan dinding, akuarium | backlight |
| 40 | wall-brick, couch, clock, shelf, toy ×5, bottle ×3, table-merged (meja bundar kecil), vase/potted plant, window-other | lukisan, bingkai | backlight |
| 41 | couch ×2, wall-brick, shelf, clock, vase, table-merged, toy, bottle, floor-other-merged | lukisan | backlight |
| 42 | couch (bangku kayu), chair (stool), window-other, floor-other-merged, wall-other-merged | mobil & motor di luar | |
| 43 | couch, chair (stool), window-other ×2, wall-brick, wall-other-merged, floor-other-merged, door-stuff | mobil di luar, lukisan, sangkar | |
| 44 | wall-other-merged, table-merged, laptop, chair, backpack, plug, paper ×3 (sticky notes), box | alas mouse | |
| 45 | wall-other-merged, cabinet-merged, bottle ×4, plug ×2, box, food-other-merged (snack) | rice cooker, pakaian, plastik, lakban | |
| 46 | cabinet-merged (lemari), shelf, pan, handbag, floor-other-merged, wall-other-merged | keranjang, plastik | |
| 47 | wall-other-merged, curtain, rug-merged (keset), floor-other-merged | keranjang cucian | |
| 48 | wall-other-merged, curtain, bed, pillow ×2 | | |
| 49 | wall-other-merged, table-merged ×2, chair ×9 (stool besi), vase ×2, door-stuff, floor-other-merged | | |
| 50 | table-merged, laptop, chair ×2, backpack/handbag, plug | topi, jalan/luar | backlight |
| 51 | person ×5, laptop ×3, tv ×4 (monitor), keyboard ×2, mouse ×2, cell phone, table-merged, chair ×3, ceiling-merged, light ×10, window-other | power bank, kabel, proyektor | **ada wajah orang** |
| 52 | chair ×12, cabinet-merged, window-other ×4, curtain, wall-other-merged, floor-other-merged, person (tangan), cell phone, door-stuff | papan/lukisan | kelas |
| 53 | person, cell phone, chair ×6, cabinet-merged, window-other, curtain, wall-other-merged, floor-other-merged | | **ada wajah orang** |
| 54 | person, chair ×3, cell phone (tablet), wall-other-merged | | **ada wajah orang** |
| 55 | chair (sangat banyak), ceiling-merged, wall-other-merged, rug-merged, door-stuff, light | AC, speaker | **sangat berat** |
| 56 | chair (sangat banyak), ceiling-merged, wall-other-merged, rug-merged, curtain (panggung), light | AC, speaker | **sangat berat**, redup |
| 57 | person, cell phone, chair ×7, counter (meja resepsionis kayu), wall-other-merged, rug-merged | | **ada wajah orang** |
| 58 | person ×2, chair ×20+, ceiling-merged, wall-other-merged, door-stuff, rug-merged | AC, vacuum cleaner | **ada wajah orang**, berat |
| 59 | chair (sangat banyak), ceiling-merged, wall-other-merged, door-stuff, rug-merged | AC, CCTV | **sangat berat** |
| 60 | ~~dihapus (resolusi berbeda)~~ | | |
| 61 | ~~dihapus (resolusi berbeda)~~ | | |

## Catatan beban kerja

- Foto ballroom **#55, #56, #59, #60, #61** masing-masing berisi puluhan sampai ratusan kursi. Satu foto bisa
  makan berjam-jam. Saran: simpan 1–2 saja (mis. #56 redup dan #60 ada orangnya), sisanya ganti dengan foto lain.
- Gudang **#20–#23** juga sangat berat; simpan 1–2.
- Foto dengan **wajah orang** (#51, #53, #54, #57, #58): pastikan orangnya setuju fotonya dipakai.
