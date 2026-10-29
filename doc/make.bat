@ECHO OFF
setlocal
REM Command file for Sphinx documentation (skrub)

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
set BUILDDIR=_build
set SOURCEDIR=.
set ALLSPHINXOPTS=-d "%BUILDDIR%/doctrees" %SPHINXOPTS% "%SOURCEDIR%"

if NOT "%PAPER%" == "" (
    set ALLSPHINXOPTS=-D latex_paper_size=%PAPER% %ALLSPHINXOPTS%
)

if NOT "%EXAMPLES_PATTERN%" == "" (
    set ALLSPHINXOPTS=-D sphinx_gallery_conf.filename_pattern=%EXAMPLES_PATTERN% %ALLSPHINXOPTS%
)

if "%1" == "" goto html-noplot
if "%1" == "help" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html
if "%1" == "html-noplot" goto html-noplot
if "%1" == "linkcheck" goto linkcheck
if "%1" == "linkcheck-noplot" goto linkcheck-noplot

echo.Unknown target "%1", use "make.bat help" for available targets.
goto end

:help
echo.Please use make.bat ^<target^> where ^<target^> is one of
echo.  html               to make standalone HTML files
echo.  html-noplot        to make HTML files without plotting the gallery
echo.  linkcheck          to check all external links for integrity
echo.  linkcheck-noplot   to check links without plotting the gallery
echo.  clean              to remove all build output
goto end

:clean
if exist "%BUILDDIR%\" (
    rmdir /q /s "%BUILDDIR%"
    echo. Removed %BUILDDIR%\
)
if exist "auto_examples\" (
    rmdir /q /s "auto_examples"
    echo. Removed auto_examples\
)
if exist "generated\" (
    rmdir /q /s "generated"
    echo. Removed generated\
)
if exist "reference\generated\" (
    rmdir /q /s "reference\generated"
    echo. Removed reference\generated\
)
if exist "generated_for_index\" (
    rmdir /q /s "generated_for_index"
    echo. Removed generated_for_index\
)
if exist "reference\" (
    for %%i in ("reference\*.rst") do del /q "%%i" 2>nul
    echo. Removed reference\*.rst
)
goto end

:html
if exist "%BUILDDIR%\html\_images\" (
    rmdir /q /s "%BUILDDIR%\html\_images"
)
set SKB_TABLE_REPORT_VERBOSITY=0
%SPHINXBUILD% -b html %ALLSPHINXOPTS% "%BUILDDIR%/html"
echo.
echo.Build finished. The HTML pages are in %BUILDDIR%/html.
goto end

:html-noplot
set SKB_TABLE_REPORT_VERBOSITY=0
%SPHINXBUILD% -D plot_gallery=0 -b html %ALLSPHINXOPTS% "%BUILDDIR%/html"
echo.
echo.Build finished. The HTML pages are in %BUILDDIR%/html.
goto end

:linkcheck
%SPHINXBUILD% -b linkcheck %ALLSPHINXOPTS% "%BUILDDIR%/linkcheck"
echo.
echo.Linkcheck finished. Results are in %BUILDDIR%/linkcheck.
goto end

:linkcheck-noplot
%SPHINXBUILD% -D plot_gallery=0 -b linkcheck %ALLSPHINXOPTS% "%BUILDDIR%/linkcheck-noplot"
echo.
echo.Linkcheck (no plot) finished. Results are in %BUILDDIR%/linkcheck-noplot.
goto end

:end
