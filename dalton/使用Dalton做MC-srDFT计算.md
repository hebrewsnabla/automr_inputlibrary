# 使用Dalton做MC-srDFT计算
## 
在`**WAVE FUNCTIONS`前增加一个section
```
**INTEGRALS
*TWOINT
.DOSRINTEGRALS
.ERF
 0.4
```

## WAVE Functions
Dalton推荐的做法是
```
.HFSRDFT
.MP2
.MCSRDFT
```
但是我们不必这么做。直接把MOKIT生成的输入文件中的`.MCSCF`改成
```
.MCSRDFT 
.SRFUN
SRXPBEHSE SRCPBERI NO_SPINDENSITY
```
如果要做NEVPT2-srDFT，再加上`.NECPT2`（实际上是用MC-srDFT轨道做的NEVPT2）。
其中`.SRFUN`的前两个参数为exchange和correlation泛函
前者可选（以下均略去LDA）
```
HFEXCH # full-range, 系数可用 .HFXFAC设置
SRXPBEGWS
SRXPBEHSE
```
后者可选
```
SRCPW92
CPW92 # full-range
SRCPBEGWS
SRCPBERI
CPBE # full-range
```
SRC开头的表示short range correlation泛函，而C开头的表示full-range correlation泛函。
内置的组合好的XC泛函包括
```
SRPBEGWS  # SRXPBEGWS, SRCPBEGWS
LRCPBEGWS # SRXPBEGWS, CPBE
SRPBE0GWS # SRXPBEGWS with 0.25 HFEXCH, SRCPBEGWS
SRPBERI   # SRXPBEGWS, SRCPBERI
```
注意这里的RI并不是resolution-of-identity，而是rational interpolation。