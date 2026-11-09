# -*- coding: utf-8 -*-
"""board_display — 완성본. 모니터에 띄우는 체커보드 (HTML) 생성.

인쇄 대신 모니터를 보드로 씁니다: 모니터는 평면이 보장되고, 칸 크기는 "칸의 픽셀 수 × 픽셀 피치" 로 정확히 정해집니다.
    from hw1lib import board_display
    board_display.write_html('../checkerboard_screen.html')     # 브라우저로 열고 F11(전체 화면)

페이지는 window.devicePixelRatio 를 보정해 칸을 '디스플레이 물리 픽셀' 단위 정수로 그리므로 macOS Retina · Windows 배율에 무관합니다.
칸 크기(mm) = 칸 픽셀 수 × 픽셀 피치(mm). 픽셀 피치 = 모니터 대각선(inch) × 25.4 / sqrt(W² + H²) (W, H = 네이티브 해상도).
"""
import os

HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>HW1 checkerboard (screen)</title>
<style>
 html,body{margin:0;background:#fff;color:#12263F;font-family:system-ui,sans-serif;overflow:hidden}
 #panel{position:fixed;left:12px;top:12px;background:rgba(255,255,255,.96);border:1px solid #D6DEE8;padding:10px 14px;font-size:13px;line-height:1.55;max-width:600px;z-index:2}
 #panel input{width:70px} #panel b{color:#1E80E1} button{margin-left:6px}
 canvas{display:block}
</style></head><body>
<div id="panel">
 <div><b>HW1 체커보드 — 10 × 7 칸 (내부 코너 9 × 6)</b> <button onclick="toggle()">패널 숨김/표시 (H)</button></div>
 <div>디스플레이 물리 해상도: <span id="res"></span> px (devicePixelRatio <span id="dpr"></span>) — 모니터 사양의 <b>네이티브 해상도</b>와 같아야 합니다. 24인치 이상 모니터 권장</div>
 <div>칸 크기(물리 픽셀): <input id="sq" type="number" min="20" step="1"> px <button onclick="draw()">적용</button> (기본값 = 화면에 맞는 최대)</div>
 <div>모니터 대각선: <input id="inch" type="number" step="0.1" placeholder="예 27"> inch → 픽셀 피치 <span id="pitch">?</span> mm → <b>칸 크기 <span id="mm">?</span> mm</b> → measurements.json 의 board_square_mm</div>
 <div style="color:#6B7280">F11 전체 화면 · 밝기 최대 · 자동 밝기 끔 · 화면 보호기 끔. 보드 아래 눈금자(1000 물리 픽셀, 100 px 간격 눈금)를 자로 재어 피치를 확인(±1 mm 이내면 OK). 촬영 전 H 로 패널을 숨깁니다.</div>
</div>
<canvas id="c"></canvas>
<script>
const NX=10, NY=7, RULER_H=130;              // RULER_H: 보드 아래 눈금자 영역(물리 px)
function dims(){const d=window.devicePixelRatio||1; return {d, W:Math.round(screen.width*d), H:Math.round(screen.height*d)};}
function defaultSq(){const {W,H}=dims(); return Math.floor(Math.min((W-80)/NX,(H-80-RULER_H)/NY));}
function draw(){
 const {d,W,H}=dims(); const c=document.getElementById('c');
 const sq=parseInt(document.getElementById('sq').value)||defaultSq();
 document.getElementById('sq').value=sq;
 c.width=Math.round(window.innerWidth*d); c.height=Math.round(window.innerHeight*d);
 c.style.width=window.innerWidth+'px'; c.style.height=window.innerHeight+'px';
 const g=c.getContext('2d'); g.fillStyle='#fff'; g.fillRect(0,0,c.width,c.height);
 const bw=NX*sq, bh=NY*sq, x0=Math.round((c.width-bw)/2), y0=Math.round((c.height-bh-RULER_H)/2);
 for(let i=0;i<NX;i++)for(let j=0;j<NY;j++){ if((i+j)%2==0){g.fillStyle='#000'; g.fillRect(x0+i*sq,y0+j*sq,sq,sq);} }
 // 눈금자: 1000 물리 픽셀, 100 px 마다 흑백 교대 (지도 축척자처럼), 끝 눈금 길게
 const rx=Math.round((c.width-1000)/2), ry=y0+bh+Math.round(RULER_H*0.3), rh=Math.round(RULER_H*0.16);
 for(let k=0;k<10;k++){ g.fillStyle=(k%2==0)?'#000':'#fff'; g.fillRect(rx+k*100, ry, 100, rh); }
 g.strokeStyle='#000'; g.lineWidth=Math.max(1,Math.round(d)); g.strokeRect(rx+0.5, ry+0.5, 1000, rh);
 for(let k=0;k<=10;k++){ const t=(k==0||k==5||k==10)?rh*0.9:rh*0.45; g.fillStyle='#000'; g.fillRect(rx+k*100-Math.round(d/2), ry-t, Math.max(1,Math.round(d)), t); }
 const inch=parseFloat(document.getElementById('inch').value);
 let pitch=NaN; if(inch>0){pitch=inch*25.4/Math.sqrt(W*W+H*H);}
 g.fillStyle='#12263F'; g.font=Math.round(RULER_H*0.17)+'px system-ui, sans-serif'; g.textAlign='center';
 g.fillText('0', rx, ry+rh+Math.round(RULER_H*0.2)); g.fillText('500', rx+500, ry+rh+Math.round(RULER_H*0.2)); g.fillText('1000 px', rx+1000, ry+rh+Math.round(RULER_H*0.2));
 g.fillStyle='#6B7280'; g.fillText('눈금자 = 1000 물리 픽셀'+(isNaN(pitch)?'':' = '+(1000*pitch).toFixed(1)+' mm') + '  — 자로 재어 ±1 mm 이내인지 확인', c.width/2, ry+rh+Math.round(RULER_H*0.42));
 document.getElementById('res').textContent=W+' × '+H; document.getElementById('dpr').textContent=d;
 document.getElementById('pitch').textContent=isNaN(pitch)?'?':pitch.toFixed(4);
 document.getElementById('mm').textContent=isNaN(pitch)?'?':(sq*pitch).toFixed(2);
}
function toggle(){const p=document.getElementById('panel'); p.style.display=(p.style.display=='none')?'block':'none';}
document.addEventListener('keydown',e=>{if(e.key=='h'||e.key=='H')toggle();});
document.getElementById('inch').addEventListener('input',draw);
window.addEventListener('resize',draw); draw();
</script></body></html>
'''


def write_html(path='checkerboard_screen.html'):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf8') as fp:
        fp.write(HTML)
    return os.path.abspath(path)


def pixel_pitch_mm(diagonal_inch, width_px, height_px):
    """픽셀 피치(mm) = 대각선(inch) × 25.4 / sqrt(W² + H²)."""
    return diagonal_inch * 25.4 / (width_px ** 2 + height_px ** 2) ** 0.5


if __name__ == '__main__':
    import sys
    print(write_html(sys.argv[1] if len(sys.argv) > 1 else 'checkerboard_screen.html'))
