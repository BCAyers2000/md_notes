# Everything compiled goes to build/; the class lives in style/.
$out_dir = 'build';
$pdf_mode = 1;
ensure_path('TEXINPUTS', './style//');
ensure_path('BSTINPUTS', './style//');
ensure_path('BIBINPUTS', './bib//');
