/* ===================================================================
   HiMoE visual essay — "pattern: alarm → resample → rescue."
   A distilled five-beat vector animation, same pure-function engine as
   the TraceGraph / NAD / RAD / ARM films: t -> frame, smootherstep
   ease, cross-faded scenes, seeded RNG, colour-as-narrative. Silent,
   deterministic, bilingual DOM captions. No video file.
     1  pattern    — the router lights 4 of 32 experts; that roster
     2  contrast   — working: reshuffles each step · stuck: frozen
     3  r(t)       — own-opening normalisation; 3 below θ → alarm
     4  crowds     — blue holds, orange sinks; 81.2% over 864 (real)
     5  fork       — save state, swap noise: ✗52 steps vs ✓8 steps
   Colour: teal=working/success · red=trap/frozen · gold=alarm/decision
   Curves are schematic; every number is measured (the captions say so).
   =================================================================== */
(function () {
  'use strict';
  var canvas = document.getElementById('film-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d', { alpha: true });
  var capEl = document.getElementById('film-caption');
  var capZh = capEl && capEl.querySelector('.film__cap-zh');
  var capEn = capEl && capEl.querySelector('.film__cap-en');
  var playBtn = document.getElementById('film-play');
  var prog = document.getElementById('film-prog');
  var stage = canvas.parentNode;
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- helpers ---- */
  function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
  function ease(t){t=t<0?0:t>1?1:t;return t*t*t*(t*(6*t-15)+10);}
  function seg(t,a,b){return b<=a?(t>=b?1:0):Math.max(0,Math.min(1,(t-a)/(b-a)));}
  function rgba(c,a){return 'rgba('+c[0]+','+c[1]+','+c[2]+','+(a<0?0:a>1?1:a)+')';}
  function lerp(a,b,u){return a+(b-a)*u;}

  var THEMES = {
    dark:  { teal:[39,227,203], gold:[255,201,91], red:[255,110,120], gray:[150,160,180], faint:[105,118,138], ink:[222,228,240] },
    light: { teal:[12,110,97],  gold:[150,100,16], red:[172,55,43],   gray:[110,110,118], faint:[140,138,128], ink:[46,59,82]   }
  };
  var theme = THEMES.dark, isDark = true;

  /* ---- stage metrics (CSS-pixel space via transform) ---- */
  var DPR=1,W=0,H=0;
  var vm=function(){return Math.min(W,H);};
  function resize(){DPR=Math.min(2,window.devicePixelRatio||1);var r=stage.getBoundingClientRect();W=Math.max(1,Math.round(r.width));H=Math.max(1,Math.round(r.height));canvas.width=Math.round(W*DPR);canvas.height=Math.round(H*DPR);ctx.setTransform(DPR,0,0,DPR,0,0);paint();}

  function glowStroke(col,w,a){
    ctx.lineJoin='round';ctx.lineCap='round';
    if(isDark){ctx.strokeStyle=rgba(col,.16*a);ctx.lineWidth=w*3.2;ctx.stroke();}
    ctx.strokeStyle=rgba(col,a);ctx.lineWidth=w;ctx.stroke();
  }
  var serif=function(px){return 'italic 500 '+px+'px "Fraunces","Noto Serif SC",Georgia,serif';};
  var sans=function(px,w){return (w||400)+' '+px+'px "Hanken Grotesk","Noto Sans SC",system-ui,sans-serif';};

  /* ---- shared bits (ported from the standalone film) ---- */
  var stars=[];
  function ringPts(cx,cy,rad){
    var P=[];for(var i=0;i<32;i++){var a=-Math.PI/2+i/32*2*Math.PI;
      P.push([cx+rad*Math.cos(a),cy+rad*Math.sin(a)]);}return P;
  }
  function makeSeq(seed,beats,swaps){
    var r=mulberry32(seed),cur=[3,11,19,27],S=[cur.slice()],b,s,n,out,c;
    for(b=0;b<beats;b++){n=cur.slice();
      for(s=0;s<swaps;s++){out=Math.floor(r()*4);
        do{c=Math.floor(r()*32);}while(n.indexOf(c)>=0);n[out]=c;}
      cur=n;S.push(cur.slice());}
    return S;
  }
  function drawRing(cx,cy,rad,seq,bt,a,col,frozenPulse){
    var P=ringPts(cx,cy,rad),i;
    for(i=0;i<32;i++){ctx.fillStyle=rgba(theme.faint,.5*a);
      ctx.beginPath();ctx.arc(P[i][0],P[i][1],rad*.028,0,7);ctx.fill();}
    var beats=seq.length-1,bi=Math.min(beats-1,Math.floor(bt)),bu=ease(bt-bi);
    var cur=seq[bi],nxt=seq[Math.min(beats,bi+1)];
    for(var s=0;s<4;s++){
      var i0=cur[s],i1=nxt[s],x,y,g=1;
      if(i0===i1){x=P[i0][0];y=P[i0][1];
        if(frozenPulse)g=1+.25*Math.sin(bt*2.2+s);}
      else{x=lerp(P[i0][0],P[i1][0],bu);y=lerp(P[i0][1],P[i1][1],bu);
        g=1+Math.sin(bu*Math.PI)*.8;}
      ctx.fillStyle=rgba(col,.13*a*g);
      ctx.beginPath();ctx.arc(x,y,rad*.12*g,0,7);ctx.fill();
      ctx.fillStyle=rgba(col,.95*a);
      ctx.beginPath();ctx.arc(x,y,rad*.05,0,7);ctx.fill();
    }
  }
  function smoothPath(pts){
    ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);
    for(var i=1;i<pts.length-1;i++){
      var xc=(pts[i][0]+pts[i+1][0])/2,yc=(pts[i][1]+pts[i+1][1])/2;
      ctx.quadraticCurveTo(pts[i][0],pts[i][1],xc,yc);}
    var l=pts[pts.length-1];ctx.lineTo(l[0],l[1]);
  }
  function wobble(x,seed){var r1=Math.sin(x*7+seed*3),r2=Math.sin(x*17+seed*7);
    return .035*r1+.02*r2;}

  /* ================= scenes (p = local progress 0..1) ================= */
  var S1D=8.5,S2D=10.5,S3D=10.0,S4D=8.0,S5D=13.0;
  var SEQ_FAST=null,SEQ_A=null;
  var zh=document.documentElement.dataset.lang!=='en';

  function s1(p,a){ /* the pulse of thinking: a live piano-roll + a beating pulse */
    if(!SEQ_FAST)SEQ_FAST=makeSeq(7,160,1);
    var NE=32;
    var yTop=H*.20,rows=H*.50,rh=rows/NE;
    var live=W*.86,cw=Math.max(9,W*.026);
    var bt=seg(p,.05,1)*17,bi=Math.floor(bt),bu=ease(bt-bi);
    var COLS=Math.floor((live-W*.05)/cw),j,s2,e;
    /* row guides at the live edge */
    var gu=ease(seg(p,0,.10));
    for(e=0;e<NE;e++){
      ctx.fillStyle=rgba(theme.faint,.35*a*gu);
      ctx.beginPath();ctx.arc(live+cw*.75,yTop+e*rh+rh*.5,1.4,0,7);ctx.fill();
    }
    /* the roll: live column bright, history streaming left */
    for(j=0;j<=COLS;j++){
      var idx=bi-j;if(idx<0)break;
      var lit=SEQ_FAST[Math.min(SEQ_FAST.length-1,idx)];
      var x=live-(j+bu)*cw;if(x<W*.03)break;
      var fade=Math.max(0,1-(j+bu)/(COLS+1));
      for(s2=0;s2<4;s2++){
        var y=yTop+lit[s2]*rh;
        var al=(j===0?.95:.5*fade)*a;
        if(isDark&&j===0){ctx.fillStyle=rgba(theme.teal,.25*a);ctx.fillRect(x-cw*.18,y-rh*.25,cw*1.1,rh*1.2);}
        ctx.fillStyle=rgba(theme.teal,al);
        ctx.fillRect(x,y,cw*.74,rh*.72);
      }
    }
    /* the pulse: one spike per swap, scrolling with the roll */
    var base=H*.815,amp=H*.055;
    var pu=ease(seg(p,.10,.22));
    if(pu>0.003){
      ctx.beginPath();var started=false;
      for(var px2=W*.03;px2<=live;px2+=3){
        var tb=bi+ (bt-bi) -(live-px2)/cw;      /* beat-time at this x */
        if(tb<0)continue;
        var k=Math.floor(tb),fr=tb-k;
        var hgt=(0.45+0.55*Math.abs(Math.sin(k*2.37+1)))*amp;
        var v=Math.pow(Math.sin(Math.min(fr,1)*Math.PI),6)*hgt;
        var y2=base-v;
        if(!started){ctx.moveTo(px2,y2);started=true;}else ctx.lineTo(px2,y2);
      }
      glowStroke(theme.teal,1.8,.8*a*pu);
    }
    ctx.font=sans(vm()*.020);ctx.textAlign='left';ctx.textBaseline='middle';
    ctx.fillStyle=rgba(theme.faint,.9*a*pu);
    ctx.fillText(zh?'每换一次手,跳一下':'one beat per reshuffle',W*.03,base-amp-vm()*.03);
    /* the name writes on, gold underline */
    var wu=ease(seg(p,.42,.58));
    if(wu>0.003){
      var wsize=vm()*.085,wy=H*.135,wx=W*.5;
      ctx.font='italic 500 '+wsize+'px "Fraunces","Noto Serif SC",Georgia,serif';
      ctx.textAlign='center';ctx.textBaseline='alphabetic';
      var wtxt='pattern',wwid=ctx.measureText(wtxt).width;
      ctx.save();ctx.beginPath();ctx.rect(wx-wwid/2-6,wy-wsize*1.1,(wwid+12)*wu,wsize*1.5);ctx.clip();
      ctx.fillStyle=rgba(theme.ink,a);ctx.fillText(wtxt,wx,wy);ctx.restore();
      var lu=ease(seg(p,.52,.66));
      if(lu>0.003){ctx.strokeStyle=rgba(theme.gold,.9*a);ctx.lineWidth=2;ctx.lineCap='round';
        ctx.beginPath();ctx.moveTo(wx-wwid/2,wy+vm()*.014);ctx.lineTo(wx-wwid/2+wwid*lu,wy+vm()*.014);ctx.stroke();}
    }
  }

  function s2(p,a){ /* two hearts: one keeps beating, one flatlines */
    if(!SEQ_FAST)SEQ_FAST=makeSeq(7,160,1);
    if(!SEQ_A)SEQ_A=makeSeq(21,160,1);
    var NE=32,freezeAt=.34;
    function panel(x0,x1,seq,frozenFrom,label,sub,subCol,btime){
      var yTop=H*.215,rows=H*.42,rh=rows/NE;
      var live=x1,cw=Math.max(7,(x1-x0)*.075);
      var bi=Math.floor(btime),bu=ease(btime-bi);
      var COLS=Math.floor((live-x0)/cw),j,s2;
      for(j=0;j<=COLS;j++){
        var idx=bi-j;if(idx<0)break;
        var frz=(frozenFrom!=null&&idx>=frozenFrom);
        var lit=seq[Math.min(seq.length-1,frz?frozenFrom:idx)];
        var x=live-(j+bu)*cw;if(x<x0)break;
        var fade=Math.max(0,1-(j+bu)/(COLS+1));
        var col=frz?theme.red:theme.teal;
        for(s2=0;s2<4;s2++){
          var y=yTop+lit[s2]*rh;
          ctx.fillStyle=rgba(col,(j===0?.95:.5*fade)*a);
          ctx.fillRect(x,y,cw*.74,rh*.72);
        }
      }
      /* pulse */
      var base=H*.76,amp=H*.045;
      ctx.beginPath();var started=false;
      for(var px2=x0;px2<=live;px2+=3){
        var tb=btime-(live-px2)/cw;if(tb<0)continue;
        var k=Math.floor(tb),fr=tb-k;
        var dead=(frozenFrom!=null&&k>=frozenFrom);
        var hgt=dead?0:(0.45+0.55*Math.abs(Math.sin(k*2.37+1)))*amp;
        var v=Math.pow(Math.sin(Math.min(fr,1)*Math.PI),6)*hgt;
        var y2=base-v;
        if(!started){ctx.moveTo(px2,y2);started=true;}else ctx.lineTo(px2,y2);
      }
      glowStroke(frozenFrom!=null?theme.red:theme.teal,1.8,.8*a);
      /* labels */
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font=sans(vm()*.030,600);
      ctx.fillStyle=rgba(frozenFrom!=null?theme.red:theme.teal,a);
      ctx.fillText(label,(x0+x1)/2,H*.145);
      ctx.font=sans(vm()*.021);
      ctx.fillStyle=rgba(subCol,a);
      ctx.fillText(sub,(x0+x1)/2,H*.845);
    }
    var born=ease(seg(p,0,.12));
    var btL=p*26;
    panel(W*.055,W*.475,SEQ_A,null,
      zh?'手臂在动':'arm working',
      zh?'pattern 一直在换':'pattern keeps changing',theme.gray,btL*born);
    var frozen=p>freezeAt;
    var btR=p*26;
    var fb=Math.floor(freezeAt*26);
    panel(W*.525,W*.945,SEQ_FAST,frozen?fb:null,
      zh?'手臂卡住':'arm stuck',
      frozen?(zh?'pattern 冻住了 —— 拉平':'pattern frozen — flatline'):(zh?'…':'…'),
      frozen?theme.red:theme.gray,btR*born);
  }

  function s3(p,a){ /* schematic r(t): flat, then a clean dive, 1-2-3, alarm */
    var x0=W*.13,x1=W*.90,y0=H*.16,y1=H*.74;
    var X=function(u){return x0+(x1-x0)*u;},Y=function(v){return y0+(y1-y0)*(1.25-v)/1.0;};
    var ax=ease(seg(p,0,.14));
    ctx.strokeStyle=rgba(theme.faint,.55*ax*a);ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(x0,y1);ctx.lineTo(x1,y1);ctx.stroke();
    ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x0,y1);ctx.stroke();
    ctx.save();ctx.translate(x0-26,(y0+y1)/2);ctx.rotate(-Math.PI/2);
    ctx.textAlign='center';ctx.font=sans(vm()*.023);
    ctx.fillStyle=rgba(theme.gray,ax*a);
    ctx.fillText(zh?'神经元 pattern':'neuron pattern',0,0);ctx.restore();
    var TH=.62,th=ease(seg(p,.10,.22));
    ctx.setLineDash([6,6]);ctx.beginPath();
    ctx.moveTo(x0,Y(TH));ctx.lineTo(x0+(x1-x0)*th,Y(TH));
    ctx.strokeStyle=rgba(theme.faint,.9*a);ctx.stroke();ctx.setLineDash([]);
    ctx.font=sans(vm()*.022);ctx.fillStyle=rgba(theme.gray,th*a);
    ctx.textAlign='left';ctx.textBaseline='bottom';
    ctx.fillText('θ',x0+8,Y(TH)-4);
    var u=ease(seg(p,.16,.66));
    var pts=[],N=140,i;
    for(i=0;i<=N*u;i++){var x=i/N;
      var v=1.0+wobble(x,1);
      if(x>.55)v-=(x-.55)/.45*.62*ease((x-.55)/.45+0.15);
      pts.push([X(x),Y(Math.max(.28,v))]);}
    if(pts.length>2){smoothPath(pts);glowStroke(theme.ink,2.2,.95*a);}
    if(u>0&&u<1){var hp=pts[pts.length-1];
      ctx.fillStyle=rgba(theme.ink,a);ctx.beginPath();ctx.arc(hp[0],hp[1],3.4,0,7);ctx.fill();}
    for(var k=0;k<3;k++){
      var xx=.80+k*.05,pu=ease(seg(p,.66+k*.06,.72+k*.06));
      if(pu<=0)continue;
      var vv=1.0+wobble(xx,1);vv-=(xx-.55)/.45*.62*ease((xx-.55)/.45+0.15);
      ctx.fillStyle=rgba(theme.gold,.95*pu*a);
      ctx.beginPath();ctx.arc(X(xx),Y(vv),3.5+2*pu,0,7);ctx.fill();
      ctx.font=sans(vm()*.020,600);ctx.textAlign='center';ctx.textBaseline='bottom';
      ctx.fillStyle=rgba(theme.gold,pu*a);
      ctx.fillText(k+1,X(xx),Y(vv)-10);
    }
    var al=seg(p,.86,1);
    if(al>0){var x2=.90;var v2=1.0+wobble(x2,1);v2-=(x2-.55)/.45*.62*ease((x2-.55)/.45+.15);
      var rr=ease(al);
      ctx.strokeStyle=rgba(theme.gold,(1-rr)*a);ctx.lineWidth=2*(1-rr)+.5;
      ctx.beginPath();ctx.arc(X(x2),Y(v2),8+50*rr,0,7);ctx.stroke();
      ctx.strokeStyle=rgba(theme.gold,a);ctx.lineWidth=2;
      ctx.beginPath();ctx.arc(X(x2),Y(v2),8,0,7);ctx.stroke();
      ctx.font=serif(vm()*.055);ctx.fillStyle=rgba(theme.gold,rr*a);
      ctx.textAlign='right';ctx.textBaseline='middle';
      ctx.fillText(zh?'报警':'alarm',X(x2)-24,Y(v2)-H*.10);}
  }

  function s4(p,a){ /* schematic split + the two real numbers */
    var x0=W*.13,x1=W*.90,y0=H*.16,y1=H*.74;
    var X=function(u){return x0+(x1-x0)*u;},Y=function(v){return y0+(y1-y0)*(1.25-v)/1.0;};
    ctx.strokeStyle=rgba(theme.faint,.5*a);ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(x0,y1);ctx.lineTo(x1,y1);ctx.stroke();
    var u=ease(seg(p,.02,.5));
    for(var i=0;i<18;i++){
      var blue=i%2===0,seed=i*1.7+2;
      var pts=[],N=110;
      for(var k=0;k<=N*u;k++){var x=k/N;
        var v=1.0+wobble(x,seed)+(blue?.04:-.02)*Math.sin(seed);
        if(!blue&&x>.35+((i%5)*.05))v-=(x-.35-((i%5)*.05))*.9;
        pts.push([X(x),Y(Math.max(.28,Math.min(1.22,v)))]);}
      if(pts.length>2){smoothPath(pts);
        ctx.strokeStyle=rgba(blue?theme.teal:theme.red,(isDark?.30:.38)*a);
        ctx.lineWidth=1.4;ctx.stroke();}
    }
    var nm=ease(seg(p,.55,.8));
    if(nm>0){
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font=serif(vm()*.13);ctx.fillStyle=rgba(theme.ink,nm*a);
      ctx.fillText('81.2%',W*.5,H*.33);
      ctx.font=sans(vm()*.022);ctx.fillStyle=rgba(theme.gray,nm*a);
      ctx.fillText(zh?'864 条真实 rollout · 在线判对':'864 real rollouts · judged online',W*.5,H*.445);
    }
  }

  function s5(p,a){ /* the fork, schematic; then the end card */
    var x0=W*.13,x1=W*.90,y0=H*.16,y1=H*.74;
    var X=function(u){return x0+(x1-x0)*u;},Y=function(v){return y0+(y1-y0)*(1.25-v)/1.0;};
    var card=seg(p,.74,.88),ax=1-ease(card);
    var FK=.42,k;
    ctx.strokeStyle=rgba(theme.faint,.5*ax*a);ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(x0,y1);ctx.lineTo(x1,y1);ctx.stroke();
    var pre=[],N=70;
    for(k=0;k<=N;k++){var x=k/N*FK;
      var v=1.0+wobble(x,1);if(x>.25)v-=(x-.25)/.17*.38*.8;
      pre.push([X(x),Y(v)]);}
    smoothPath(pre);glowStroke(theme.ink,2.0,.8*ax*a);
    var fy=pre[pre.length-1][1];
    ctx.strokeStyle=rgba(theme.gold,.9*ax*a);ctx.lineWidth=1.8;
    ctx.beginPath();ctx.arc(X(FK),fy,7,0,7);ctx.stroke();
    var uo=ease(seg(p,.06,.55));
    var po=[];for(k=0;k<=90*uo;k++){var xo=FK+k/90*(1-FK);
      var vo=(1.25-(fy-y0)/(y1-y0))-((xo-FK)/(1-FK))*.22+wobble(xo,3)*.5;
      po.push([X(xo),Y(Math.max(.30,vo))]);}
    if(po.length>2){smoothPath(po);glowStroke(theme.red,2.2,.9*ax*a);}
    var ub=ease(seg(p,.12,.48));
    var endX=FK+.17;
    var pb=[];for(k=0;k<=70*ub;k++){var xb=FK+k/70*(endX-FK);
      var rise=ease((xb-FK)/(endX-FK));
      var vb=(1.25-(fy-y0)/(y1-y0))+rise*.42+wobble(xb,5)*.4;
      pb.push([X(xb),Y(Math.min(1.2,vb))]);}
    if(pb.length>2){smoothPath(pb);glowStroke(theme.teal,2.6,a*ax);}
    if(ub>=1){
      var e=pb[pb.length-1];
      ctx.fillStyle=rgba(theme.teal,ax*a);ctx.beginPath();ctx.arc(e[0],e[1],5,0,7);ctx.fill();
      ctx.strokeStyle=rgba(theme.teal,ax*a);ctx.lineWidth=2.4;ctx.lineCap='round';
      ctx.beginPath();ctx.moveTo(e[0]+14,e[1]-2);ctx.lineTo(e[0]+20,e[1]+5);ctx.lineTo(e[0]+32,e[1]-10);ctx.stroke();
      ctx.font=sans(vm()*.024,600);ctx.fillStyle=rgba(theme.teal,ax*a);
      ctx.textAlign='left';ctx.textBaseline='bottom';
      ctx.fillText(zh?'8 步 · 完成':'8 steps · done',e[0]+42,e[1]+8);
    }
    if(uo>=1){
      var eo=po[po.length-1];
      ctx.strokeStyle=rgba(theme.red,ax*a);ctx.lineWidth=2.4;ctx.lineCap='round';
      ctx.beginPath();ctx.moveTo(eo[0]+12,eo[1]-8);ctx.lineTo(eo[0]+26,eo[1]+6);
      ctx.moveTo(eo[0]+26,eo[1]-8);ctx.lineTo(eo[0]+12,eo[1]+6);ctx.stroke();
      ctx.font=sans(vm()*.024,600);ctx.fillStyle=rgba(theme.red,ax*a);
      ctx.textAlign='right';ctx.textBaseline='top';
      ctx.fillText(zh?'52 步 · 失败':'52 steps · failed',eo[0],eo[1]+14);
    }
    var an=ease(seg(p,.02,.16));
    ctx.font=sans(vm()*.021);ctx.fillStyle=rgba(theme.gold,an*ax*a);
    ctx.textAlign='center';ctx.textBaseline='bottom';
    ctx.fillText(zh?'存状态 · 换噪声':'save state · swap the noise',X(FK),fy-18);
    if(card>0){var cu=ease(card);
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font=serif(vm()*.070);ctx.fillStyle=rgba(theme.ink,cu*a);
      ctx.fillText(zh?'报警 → 干预 → 救回':'alarm → intervene → rescue',W*.5,H*.40);
      ctx.font=serif(vm()*.038);ctx.fillStyle=rgba(theme.gray,cu*a);
      ctx.fillText('26.0 s  →  3.9 s',W*.5,H*.53);}
  }

  /* ---- assembly + engine ---- */
  var FADE=.9;
  var STORY=[[s1,S1D],[s2,S2D],[s3,S3D],[s4,S4D],[s5,S5D]];
  var T=STORY.reduce(function(s,x){return s+x[1];},0);   // 50 s
  function draw(t){
    ctx.clearRect(0,0,W,H);
    ctx.save();ctx.globalCompositeOperation=isDark?'lighter':'source-over';
    for(var s=0;s<stars.length;s++){var st=stars[s];var al=(isDark?0.05:0.022)+0.02*Math.sin(t*0.5+st.ph);
      if(al>0){ctx.beginPath();ctx.arc(st.x*W,st.y*H,st.rr,0,7);ctx.fillStyle=rgba(isDark?[210,225,245]:theme.gray,al);ctx.fill();}}
    ctx.restore();
    var acc=0;
    for(var i=0;i<STORY.length;i++){
      var fn=STORY[i][0],dur=STORY[i][1],lo=acc,hi=acc+dur;acc=hi;
      if(t<lo-FADE||t>hi+FADE)continue;
      var pp=Math.max(0,Math.min(1,(t-lo)/dur));
      var alpha=Math.min(seg(t,lo-FADE*.2,lo+FADE),1-seg(t,hi-FADE*.3,hi+FADE*.7));
      if(alpha<=0)continue;
      fn(pp,ease(alpha));
    }
  }
  function paint(){draw((!playing&&t===0&&!ended)?33.0:t);}

  /* ---- captions (zh + en spans; data-lang picks one) ---- */
  var BEATS=[
    {t:0.0, zh:'每个控制步,都有一小撮神经元被<b>激活</b>;亮起的组合,一步一变 —— 思考在流动。',
            en:'Every control step <b>activates</b> a small set of neurons; the lit combination shifts step by step — thought in flow.'},
    {t:8.5, zh:'手臂干活,流动不停;手臂一卡,<b>思考凝固</b> —— 同一撮神经元,一亮亮到超时。',
            en:'While the arm works, the flow never stops; the moment it stalls, <b>thought freezes</b> — the same few neurons lit through to timeout.'},
    {t:19.0,zh:'给流动测速:连续三步跌破 θ —— <b style="color:var(--gold)">报警</b>。',
            en:'Clock the flow: three steps below θ — <b style="color:var(--gold)">alarm</b>.'},
    {t:29.0,zh:'放到全体:蓝色浮着,橙色沉底。864 条真实 rollout、同一套常数:判对 <b>81.2%</b>。',
            en:'Zoom out: blue floats, orange sinks. 864 real rollouts, one frozen set of constants: <b>81.2%</b> correct.'},
    {t:37.0,zh:'报警那一步:<b>冻结状态,加一点干扰,继续。</b>原噪声跑满 52 步失败;干预之后 <b>8 步完成</b>。',
            en:'At the alarm: <b>freeze the state, inject a little perturbation, continue.</b> The original noise fails in 52 steps; after the nudge, <b>done in 8</b>.'},
    {t:48.0,zh:'<b>报警 → 干预 → 救回。</b>26.0 s 的失败,变成 3.9 s 的成功。<span style="font-size:.8em;opacity:.75">(曲线为示意;所有数字为实测)</span>',
            en:'<b>Alarm → intervene → rescue.</b> A 26.0 s failure becomes a 3.9 s success. <span style="font-size:.8em;opacity:.75">(curves schematic; every number measured)</span>'}
  ];
  var curBeat=-1;
  function caption(t){var bi=0;for(var i=0;i<BEATS.length;i++)if(t>=BEATS[i].t)bi=i;
    if(bi===curBeat||!capEl)return;curBeat=bi;capEl.classList.remove('show');
    setTimeout(function(){if(capZh)capZh.innerHTML=BEATS[bi].zh;if(capEn)capEn.innerHTML=BEATS[bi].en;capEl.classList.add('show');},180);}

  /* ---- playback ---- */
  var t=0,playing=false,raf=0,last=0,ended=false;
  function tick(now){if(!playing)return;if(!last)last=now;var dt=Math.min(0.05,(now-last)/1000);last=now;t+=dt;
    if(t>=T){t=T-0.04;playing=false;ended=true;setBtn();}
    draw(t);caption(t);if(prog)prog.style.width=(t/T*100)+'%';if(playing)raf=requestAnimationFrame(tick);}
  function play(){if(playing)return;if(ended){t=0;ended=false;curBeat=-1;}playing=true;last=0;setBtn();raf=requestAnimationFrame(tick);}
  function pause(){playing=false;cancelAnimationFrame(raf);setBtn();}
  function setBtn(){if(!playBtn)return;playBtn.classList.toggle('is-hidden',playing);playBtn.setAttribute('aria-label',ended?'Replay':(playing?'Pause':'Play'));playBtn.innerHTML=ended?'↻':'▶';}

  /* ---- wire up ---- */
  isDark=document.documentElement.dataset.theme!=='light';
  theme=isDark?THEMES.dark:THEMES.light;
  (function(){var r=mulberry32(11);stars=[];for(var i=0;i<52;i++)stars.push({x:r(),y:r(),rr:r()*1.0+0.3,ph:r()*6.28});})();
  if(document.fonts&&document.fonts.load){try{document.fonts.load('600 1em Fraunces');document.fonts.load('italic 1em Fraunces');}catch(e){}}
  resize();caption(0);paint();

  window.radFilm={
    setTheme:function(name){isDark=(name!=='light');theme=isDark?THEMES.dark:THEMES.light;paint();},
    seek:function(x){playing=false;cancelAnimationFrame(raf);t=Math.max(0,Math.min(T-0.04,x));ended=(t>=T-0.05);paint();caption(t);if(prog)prog.style.width=(t/T*100)+'%';setBtn();}
  };

  var rz;window.addEventListener('resize',function(){clearTimeout(rz);rz=setTimeout(resize,150);},{passive:true});
  if(playBtn)playBtn.addEventListener('click',function(e){e.stopPropagation();playing?pause():play();});
  canvas.addEventListener('click',function(){playing?pause():play();});
  document.addEventListener('visibilitychange',function(){if(document.hidden&&playing)pause();});
  if(!reduce&&'IntersectionObserver' in window){var seen=false;new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting&&!seen){seen=true;play();}else if(!e.isIntersecting&&playing){pause();}});},{threshold:0.45}).observe(stage);}
  else if(reduce){draw(T-1);caption(T-1);}
  setBtn();
})();
