/* ===================================================================
   HiMoE hero — "an arm works, freezes, and is rescued."
   Left: a schematic robot arm over a small table scene — it works,
   stalls mid-air (frost, tremble, red), three gold ticks, an alarm,
   a fresh noise flies in, it thaws, then picks the pot and sets it
   into the bowl. Right: the routing stream — 32 expert rows, the
   lit-4 roster flowing in column by column; freezing draws stripes.
   Loops every 26 s.
     • day   → ink engraving (open joints, thin strokes, dashed frost)
     • night → luminous constellation (additive glow, ignition halos)
   Same engine family as the TraceGraph / NAD / ARM heroes. Canvas 2D.
   =================================================================== */
(function () {
  'use strict';
  var canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d', { alpha: true });

  function rgba(c,a){return 'rgba('+c[0]+','+c[1]+','+c[2]+','+(a<0?0:a>1?1:a)+')';}
  function lerpC(a,b,u){return [a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u,a[2]+(b[2]-a[2])*u];}
  function ease(t){t=t<0?0:t>1?1:t;return t*t*t*(t*(6*t-15)+10);}
  function seg(t,a,b){return b<=a?(t>=b?1:0):Math.max(0,Math.min(1,(t-a)/(b-a)));}
  function clamp(v,a,b){return v<a?a:v>b?b:v;}
  function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

  var THEMES = {
    dark:  { teal:[39,227,203], gold:[255,201,91], green:[73,227,155], red:[255,110,120], gray:[126,139,162], ink:[233,227,212] },
    light: { teal:[12,110,97],  gold:[150,100,16], green:[27,122,68],  red:[172,55,43],   gray:[120,114,102], ink:[29,26,20] }
  };
  var DPR=1,W=0,H=0;
  var theme=THEMES.light,isDark=false;
  var reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
  var px={x:0,y:0},pxT={x:0,y:0};
  var running=false,raf=0,t0=0,tSec=0;

  /* node-space -> screen (edges boxed so the hero copy stays clear; stacks on mobile) */
  function MX(nx){return (0.02+nx*0.95)*W;}
  function MY(ny){var band=Math.min(H,W*0.60);var base=(W<760*DPR)?H*0.80:H*0.50;return base+(ny-0.56)*band;}

  var stars=[];
  (function(){var sr=mulberry32(17);for(var i=0;i<80;i++)stars.push({x:sr(),y:sr(),r:sr()*1.0+0.25,ph:sr()*6.28});})();

  /* ---- the loop ---- */
  var P=26;
  function freezeU(ph){return ease(seg(ph,9.4,11.0))*(1-ease(seg(ph,16.2,17.0)));}

  /* end-effector path keyframes, in arm units relative to the base
     (x right, y DOWN — negative y is up).  [phase, x, y, gripDir]
     gripDir ≈ +1.6..+2.6 keeps the hand pointing downward. */
  var POT=[-1.30,-0.075], BOWL=[-0.80,-0.055];
  var KEY=[
    [ 0.0, -0.34, -1.00,  2.50],      /* home */
    [ 1.0,  0.28, -0.28,  1.72],      /* dart right, over the cup */
    [ 1.35, 0.30, -0.11,  1.62],      /* tap */
    [ 1.6,  0.30, -0.24,  1.64],
    [ 1.85, 0.30, -0.12,  1.62],      /* tap again */
    [ 2.15, 0.24, -0.36,  1.78],
    [ 3.0, -0.80, -0.34,  1.85],      /* swing over the bowl */
    [ 3.2, -0.80, -0.17,  1.80],      /* lower in — stir */
    [ 4.8, -0.80, -0.19,  1.80],
    [ 5.3, -0.74, -0.52,  2.20],      /* raise + shake the wrist */
    [ 5.65,-0.78, -0.48,  2.66],
    [ 6.0, -0.72, -0.50,  2.06],
    [ 6.35,-0.76, -0.49,  2.48],
    [ 6.9, -0.42, -0.86,  2.45],      /* retreat, hover */
    [ 7.6, -0.50, -0.80,  2.40],
    [ 9.0, -0.64, -0.68,  2.30],      /* heading for the pot… */
    [11.0, -0.68, -0.64,  2.28],      /* …frozen mid-errand */
    [17.0, -0.68, -0.64,  2.28],
    [19.2, -1.30, -0.22,  1.95],      /* finish the reach — onto the pot */
    [20.1, -1.30, -0.22,  1.95],      /* dwell — grab */
    [21.6, -1.05, -0.75,  2.05],      /* lift + carry */
    [23.0, -0.80, -0.30,  1.85],      /* lower into the bowl */
    [23.6, -0.80, -0.30,  1.85],      /* dwell — release */
    [24.3, -0.92, -0.58,  2.05],      /* pull back, a look at the bowl */
    [25.2, -0.46, -0.92,  2.40],      /* retract */
    [26.0, -0.34, -1.00,  2.50]
  ];
  function poseAt(ph){
    var i=0;while(i<KEY.length-2&&ph>=KEY[i+1][0])i++;
    var k0=KEY[i],k1=KEY[i+1],u=ease(seg(ph,k0[0],k1[0]));
    return [k0[1]+(k1[1]-k0[1])*u, k0[2]+(k1[2]-k0[2])*u, k0[3]+(k1[3]-k0[3])*u];
  }

  /* ---- draw helpers ---- */
  function gdot(x,y,r,col,a){if(a<=0.003)return;var Pt=[[r*2.6,0.10],[r*1.6,0.22],[r,1]];for(var p=0;p<Pt.length;p++){ctx.beginPath();ctx.arc(x,y,Pt[p][0],0,6.2832);ctx.fillStyle=rgba(col,a*Pt[p][1]);ctx.fill();}}
  function disc(x,y,r,col,a){if(a<=0.003)return;ctx.beginPath();ctx.arc(x,y,r,0,6.2832);ctx.fillStyle=rgba(col,a);ctx.fill();}
  function circle(x,y,r,col,a,w){if(a<=0.003)return;ctx.beginPath();ctx.arc(x,y,r,0,6.2832);ctx.lineWidth=w;ctx.strokeStyle=rgba(col,a);ctx.stroke();}
  function line(x0,y0,x1,y1,col,a,w){if(a<=0.003)return;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.lineCap='round';ctx.lineWidth=w;ctx.strokeStyle=rgba(col,a);ctx.stroke();}
  function ringS(x,y,r,col,a,w,dash){if(a<=0.003||r<=0)return;ctx.save();if(dash)ctx.setLineDash([4*DPR,6*DPR]);ctx.beginPath();ctx.arc(x,y,r,0,6.2832);ctx.lineWidth=w;ctx.strokeStyle=rgba(col,a);ctx.stroke();ctx.setLineDash([]);ctx.restore();}
  function arc(x,y,r,a0,a1,col,a,w){if(a<=0.003)return;ctx.beginPath();ctx.arc(x,y,r,a0,a1);ctx.lineWidth=w;ctx.lineCap='round';ctx.strokeStyle=rgba(col,a);ctx.stroke();}
  /* a link drawn as a small truss: two parallel rails + end caps */
  function truss(x0,y0,x1,y1,col,a,w,off){
    var dx=x1-x0,dy=y1-y0,L=Math.sqrt(dx*dx+dy*dy)||1;
    var nx=-dy/L*off,ny=dx/L*off;
    if(isDark)line(x0,y0,x1,y1,col,0.20*a,w*3.4+off*2);
    line(x0+nx,y0+ny,x1+nx,y1+ny,col,a,w);
    line(x0-nx,y0-ny,x1-nx,y1-ny,col,a,w);
    line(x0+nx,y0+ny,x0-nx,y0-ny,col,a,w);
    line(x1+nx,y1+ny,x1-nx,y1-ny,col,a,w);
  }
  function motor(x,y,r,col,a,w){                      // joint "motor" disc
    if(isDark){gdot(x,y,r*0.55,col,0.9*a);circle(x,y,r,col,0.55*a,w);}
    else{circle(x,y,r,col,0.85*a,w);circle(x,y,r*0.55,col,0.6*a,w*0.85);disc(x,y,r*0.16,col,a);}
  }

  function geom(){
    var mob=(W<760*DPR);
    var band=Math.min(H,W*0.60);
    var rad=Math.min(band*0.40,H*0.38,W*(mob?0.26:0.145));
    /* the table spans bx-1.62r … bx+0.55r — keep all of it on screen */
    var bx=mob?(W*0.5+0.535*rad):Math.max(W*0.03+1.64*rad,MX(0.19));
    var by=MY(0.56)+rad*0.72;
    return {bx:bx,by:by,rad:rad,mob:mob};
  }

  /* 2-link IK, elbow-up */
  function ik(bx,by,tx,ty,L1,L2){
    var dx=tx-bx,dy=ty-by,d=Math.sqrt(dx*dx+dy*dy);
    d=clamp(d,Math.abs(L1-L2)+1e-3,L1+L2-1e-3);
    var ang=Math.atan2(dy,dx);
    var cosA=clamp((L1*L1+d*d-L2*L2)/(2*L1*d),-1,1);
    var sh=ang-Math.acos(cosA);
    return [bx+L1*Math.cos(sh),by+L1*Math.sin(sh)];
  }

  /* ---- the routing stream (right side) ---- */
  var NE=32,SEQ=[];
  (function(){var r=mulberry32(7),cur=[3,11,19,27];SEQ.push(cur.slice());
    for(var b=0;b<640;b++){var n=cur.slice();var sw=1+(b%3===0?1:0);
      for(var s=0;s<sw;s++){var out=Math.floor(r()*4),c;
        do{c=Math.floor(r()*NE);}while(n.indexOf(c)>=0);n[out]=c;}
      cur=n;SEQ.push(cur.slice());}})();
  var RATE=2.2;
  function bAt(ph){
    if(ph<9)return RATE*ph;
    if(ph<11){var u=ph-9;return RATE*9+RATE*(u-u*u/4);}
    if(ph<16.7)return RATE*10;
    return RATE*10+RATE*(ph-16.7);
  }
  /* the "mind": 32 neurons in an organic cloud (brain-like, not a brain),
     plus the thinking stream pouring out of it */
  var CORE=[],CEDGE=[];
  (function(){
    var r=mulberry32(23),rings=[[5,0.34],[11,0.68],[16,1.0]],i,k;
    for(k=0;k<rings.length;k++){var n=rings[k][0],rr=rings[k][1];
      for(i=0;i<n;i++){var a=i/n*6.2832+r()*0.5,j2=rr*(1+(r()-0.5)*0.24);
        CORE.push([Math.cos(a)*j2*1.18,Math.sin(a)*j2*0.88]);}}
    for(i=0;i<CORE.length;i++){                       // each node → its 2 nearest
      var ds=[];
      for(k=0;k<CORE.length;k++){if(k===i)continue;
        var dx=CORE[i][0]-CORE[k][0],dy=CORE[i][1]-CORE[k][1];
        ds.push([dx*dx+dy*dy,k]);}
      ds.sort(function(a,b){return a[0]-b[0];});
      for(k=0;k<2;k++){var j3=ds[k][1];
        if(i<j3)CEDGE.push([i,j3]);else CEDGE.push([j3,i]);}
    }
    CEDGE=CEDGE.filter(function(e,idx){for(var q=0;q<idx;q++)if(CEDGE[q][0]===e[0]&&CEDGE[q][1]===e[1])return false;return true;});
  })();

  function mindStream(ph){
    var band=Math.min(H,W*0.60);
    var ccx=MX(0.815),ccy=MY(0.285),cr=band*0.115;
    var lit=SEQ[Math.min(SEQ.length-1,Math.floor(bAt(ph)))];
    var fz=freezeU(ph),col=lerpC(theme.teal,theme.red,fz);
    var pulse=0.5+0.5*Math.sin(tSec*(fz>0.5?1.1:2.6));
    var i,s,x,y;

    /* faint halo behind the cloud */
    if(isDark)gdot(ccx,ccy,cr*0.7,col,0.06);
    else circle(ccx,ccy,cr*1.22,theme.gray,0.14,1*DPR);

    /* edges */
    for(i=0;i<CEDGE.length;i++){
      var A=CORE[CEDGE[i][0]],B=CORE[CEDGE[i][1]];
      line(ccx+A[0]*cr,ccy+A[1]*cr,ccx+B[0]*cr,ccy+B[1]*cr,theme.gray,isDark?0.13:0.22,1*DPR);
    }
    /* the current thought: a faint loop through the 4 lit neurons */
    for(s=0;s<4;s++){
      var A2=CORE[lit[s]],B2=CORE[lit[(s+1)%4]];
      line(ccx+A2[0]*cr,ccy+A2[1]*cr,ccx+B2[0]*cr,ccy+B2[1]*cr,col,(isDark?0.20:0.30)*(0.6+0.4*pulse),1.1*DPR);
    }
    /* neurons */
    for(i=0;i<NE;i++){
      x=ccx+CORE[i][0]*cr;y=ccy+CORE[i][1]*cr;
      var isLit=lit.indexOf(i)>=0;
      if(isLit){
        if(isDark){gdot(x,y,cr*0.055*(1+0.25*pulse),col,0.95);}
        else{disc(x,y,cr*0.045,col,0.95);circle(x,y,cr*0.085*(1+0.2*pulse),col,0.6,1.1*DPR);}
      }else{
        var tw=0.5+0.5*Math.sin(tSec*1.3+i*2.1);
        if(isDark)gdot(x,y,cr*0.026,theme.gray,0.28+0.14*tw);
        else disc(x,y,cr*0.026,theme.gray,0.42+0.15*tw);
      }
    }

    /* the thinking stream below */
    var x0=MX(0.615),x1=MX(0.965),yTop=MY(0.52),yBot=MY(0.90);
    var COLS=24,sx=(x1-x0)/COLS,sy=(yBot-yTop)/(NE-1);
    var dt=0.30;
    for(var j=0;j<COLS;j++){
      var tj=ph-j*dt;if(tj<0)tj+=P;
      var lj=SEQ[Math.min(SEQ.length-1,Math.floor(bAt(tj)))];
      var fzj=freezeU(tj),colr=lerpC(theme.teal,theme.red,fzj);
      var fade=1-j/(COLS+2);
      x=x1-j*sx;
      for(s=0;s<4;s++){
        y=yTop+lj[s]*sy;
        if(isDark)gdot(x,y,Math.max(1.6*DPR,sx*0.14),colr,0.5*fade);
        else disc(x,y,Math.max(1.5*DPR,sx*0.13),colr,0.75*fade);
      }
    }
    /* funnel: the newest thoughts pour from the cloud into the stream */
    for(s=0;s<4;s++){
      var A3=CORE[lit[s]];
      var fx=ccx+A3[0]*cr,fy=ccy+A3[1]*cr;
      var tx=x1,ty=yTop+lit[s]*sy;
      ctx.beginPath();ctx.moveTo(fx,fy);
      ctx.quadraticCurveTo(ccx+cr*0.3,(fy+ty)/2+cr*0.2,tx,ty);
      ctx.lineWidth=1*DPR;ctx.strokeStyle=rgba(col,(isDark?0.12:0.18)*(0.5+0.5*pulse));ctx.stroke();
    }
    /* alarm: gold flash through the mind + a sweep down the stream */
    var al=seg(ph,14.6,15.6);
    if(al>0&&al<1){var au=ease(al);
      ringS(ccx,ccy,cr*(0.4+0.9*au),theme.gold,(1-au)*(isDark?0.6:0.8),1.6*DPR,false);
      line(x1+sx*0.2,yTop-sy,x1+sx*0.2,yBot+sy,theme.gold,(1-au)*0.6,2*DPR);}
  }

  /* ---- the arm + table scene ---- */
  function scene(ph){
    var g=geom(),rad=g.rad,bx=g.bx,by=g.by;
    if(!g.mob)mindStream(ph);

    var fz=freezeU(ph);
    var col=lerpC(theme.teal,theme.red,fz);
    var lw=Math.max(2.0*DPR,rad*0.016);
    var pc2=isDark?theme.gray:theme.ink;

    /* pose + tremble while frozen */
    var po=poseAt(ph);
    var tr=fz*(1-ease(seg(ph,15.2,16.6)));
    var ex=bx+po[0]*rad+Math.sin(tSec*21)*rad*0.008*tr;
    var ey=by+po[1]*rad+Math.cos(tSec*17)*rad*0.008*tr;
    var grip=po[2]+0.05*Math.sin(ph*2.7)*(1-fz);       // wrist micro-sway
    /* stirring inside the bowl: a slow circle overlaid on the pose */
    var su=ease(seg(ph,3.15,3.5))*(1-ease(seg(ph,4.5,4.85)));
    if(su>0.01){var sa=(ph-3.15)*5.2;
      ex+=Math.cos(sa)*rad*0.05*su;ey+=Math.sin(sa)*rad*0.028*su;
      grip+=Math.cos(sa)*0.10*su;}

    /* table */
    var gy=by+rad*0.035;
    line(bx-rad*1.62,gy,bx+rad*0.55,gy,pc2,isDark?0.28:0.4,1.2*DPR);
    line(bx-rad*1.46,gy,bx-rad*1.46,gy+rad*0.10,pc2,isDark?0.18:0.28,1.1*DPR);
    line(bx+rad*0.40,gy,bx+rad*0.40,gy+rad*0.10,pc2,isDark?0.18:0.28,1.1*DPR);

    /* the bowl — destination; wobbles + ripples while being stirred */
    var stirU=ease(seg(ph,3.15,3.5))*(1-ease(seg(ph,4.5,4.85)));
    var stirA=(ph-3.15)*5.2;
    var bwx=bx+BOWL[0]*rad,bwy=gy,bwr=rad*0.115;
    ctx.save();
    if(stirU>0.01){ctx.translate(bwx,gy);ctx.rotate(0.05*Math.sin(stirA)*stirU);ctx.translate(-bwx,-gy);}
    arc(bwx,bwy,bwr,0.15,Math.PI-0.15,pc2,isDark?0.55:0.75,1.5*DPR);
    line(bwx-bwr*0.55,bwy+bwr*0.75,bwx+bwr*0.55,bwy+bwr*0.75,pc2,isDark?0.4:0.55,1.3*DPR);
    if(isDark)gdot(bwx,bwy+bwr*0.3,bwr*0.3,pc2,0.10);
    ctx.restore();
    if(stirU>0.01){                              // ripples spreading in the bowl
      for(var rp=0;rp<2;rp++){
        var fr=((ph-3.2)/0.7+rp*0.5)%1;
        arc(bwx,bwy-bwr*0.06,bwr*(0.30+0.55*fr),Math.PI*0.15,Math.PI*0.85,col,(1-fr)*0.5*stirU,1.1*DPR);
      }
    }

    /* the cup — hops and tilts when tapped, gets nudged aside */
    var tap1=Math.sin(Math.PI*seg(ph,1.28,1.55)),tap2=Math.sin(Math.PI*seg(ph,1.78,2.05));
    var nudge=(ease(seg(ph,1.34,1.6))*0.35+ease(seg(ph,1.84,2.1))*0.65)*(1-ease(seg(ph,25.3,25.95)));
    var cx2=bx+rad*0.30+rad*0.035*nudge,cr=rad*0.05;
    var hop=rad*0.030*(tap1+tap2);
    var tilt=0.20*tap1-0.26*tap2;
    ctx.save();
    ctx.translate(cx2,gy);ctx.rotate(tilt);ctx.translate(-cx2,-gy);
    line(cx2-cr,gy-hop,cx2-cr*0.7,gy-cr*1.5-hop,pc2,isDark?0.35:0.5,1.2*DPR);
    line(cx2+cr,gy-hop,cx2+cr*0.7,gy-cr*1.5-hop,pc2,isDark?0.35:0.5,1.2*DPR);
    line(cx2-cr*0.7,gy-cr*1.5-hop,cx2+cr*0.7,gy-cr*1.5-hop,pc2,isDark?0.35:0.5,1.2*DPR);
    ctx.restore();

    /* the pot: home → carried → in the bowl → fades, loops */
    var carried=ph>=19.9&&ph<23.4;
    var inBowl=ph>=23.4&&ph<25.4;
    var potA=1;
    if(ph>=25.4)potA=1-ease(seg(ph,25.4,25.95));
    if(ph<0.7)potA=ease(seg(ph,0.05,0.7));
    var potR=rad*0.07;
    var pxy;
    if(carried)pxy=[ex,ey+rad*0.085];
    else if(inBowl)pxy=[bwx,bwy-potR*0.55];
    else pxy=[bx+POT[0]*rad,gy-potR];
    if(potA>0.01){
      var pc=carried?theme.green:(inBowl?theme.green:pc2);
      var potTilt=0;
      if(ph>=19.35&&ph<20.1)potTilt=0.10*Math.sin((ph-19.35)*16)*(1-ease(seg(ph,19.9,20.1)));   // rocks under the grip
      else if(carried)potTilt=0.08*Math.sin(tSec*6);                                             // sways in the hand
      else if(inBowl)potTilt=0.14*Math.sin((ph-23.4)*10)*(1-ease(seg(ph,23.4,24.2)));            // settles into the bowl
      ctx.save();
      if(potTilt){ctx.translate(pxy[0],pxy[1]+potR);ctx.rotate(potTilt);ctx.translate(-pxy[0],-pxy[1]-potR);}
      if(isDark){gdot(pxy[0],pxy[1],potR*0.55,pc,0.75*potA);circle(pxy[0],pxy[1],potR,pc,0.5*potA,1.1*DPR);}
      else{circle(pxy[0],pxy[1],potR,pc,0.8*potA,1.3*DPR);disc(pxy[0],pxy[1],potR*0.3,pc,0.6*potA);}
      line(pxy[0]+potR,pxy[1]-potR*0.35,pxy[0]+potR*1.6,pxy[1]-potR*0.55,pc,0.6*potA,1.1*DPR);
      ctx.restore();
    }
    /* grab + release flashes */
    var gf=seg(ph,19.6,20.6);
    if(gf>0&&gf<1)ringS(ex,ey,potR*(1+1.6*ease(gf)),theme.green,(1-ease(gf))*0.8,1.5*DPR,false);
    var rf=seg(ph,23.4,24.4);
    if(rf>0&&rf<1)ringS(bwx,bwy-bwr*0.2,bwr*(1+0.9*ease(rf)),theme.green,(1-ease(rf))*0.8,1.5*DPR,false);

    /* trail (night only, while working) */
    if(isDark&&fz<0.3){
      for(var k=1;k<=10;k++){
        var p2=ph-k*0.09;if(p2<0)p2+=P;
        var pt=poseAt(p2);
        gdot(bx+pt[0]*rad,by+pt[1]*rad,rad*0.010,theme.teal,0.10*(1-k/11)*(1-fz));
      }
    }

    /* IK chain: base → elbow → wrist → tool tip */
    var L1=rad*0.62,L2=rad*0.55,L3=rad*0.20;
    var wx=ex-L3*Math.cos(grip),wy=ey-L3*Math.sin(grip);
    var J=ik(bx,by,wx,wy,L1,L2);

    /* pedestal: plate + trapezoid tower + bolts */
    line(bx-rad*0.22,gy,bx+rad*0.22,gy,pc2,isDark?0.6:0.85,lw*1.25);
    line(bx-rad*0.15,gy,bx-rad*0.075,by+rad*0.02,pc2,isDark?0.5:0.7,lw);
    line(bx+rad*0.15,gy,bx+rad*0.075,by+rad*0.02,pc2,isDark?0.5:0.7,lw);
    line(bx-rad*0.075,by+rad*0.02,bx+rad*0.075,by+rad*0.02,pc2,isDark?0.5:0.7,lw);
    if(!isDark){disc(bx-rad*0.185,gy-rad*0.012,1.6*DPR,pc2,0.8);disc(bx+rad*0.185,gy-rad*0.012,1.6*DPR,pc2,0.8);}

    /* links as trusses + motors at the joints */
    truss(bx,by,J[0],J[1],col,0.92,lw*0.8,rad*0.022);
    truss(J[0],J[1],wx,wy,col,0.92,lw*0.75,rad*0.017);
    line(wx,wy,ex,ey,col,0.92,lw*0.85);
    motor(bx,by,rad*0.055,fz>0.5?col:pc2,1,lw*0.8);
    motor(J[0],J[1],rad*0.042,col,1,lw*0.75);
    motor(wx,wy,rad*0.030,col,1,lw*0.7);

    /* gripper: palm bar + two L-fingers, closed while carrying */
    var gdx=Math.cos(grip),gdy=Math.sin(grip);
    var pnx=-gdy,pny=gdx;                       // palm normal
    var palm=rad*0.075,fl=rad*0.115,fin=rad*0.055;
    var spread=carried?0.55:1.0;
    line(ex-pnx*palm*spread,ey-pny*palm*spread,ex+pnx*palm*spread,ey+pny*palm*spread,col,0.92,lw*0.8);
    for(var s2=-1;s2<=1;s2+=2){
      var fx0=ex+pnx*palm*spread*s2,fy0=ey+pny*palm*spread*s2;
      var fx1=fx0+gdx*fl,fy1=fy0+gdy*fl;
      line(fx0,fy0,fx1,fy1,col,0.92,lw*0.75);
      line(fx1,fy1,fx1-pnx*fin*s2,fy1-pny*fin*s2,col,0.92,lw*0.7);
    }
    if(isDark)gdot(ex,ey,rad*0.016,col,0.8);

    /* frost ring while frozen */
    var frost=ease(seg(ph,10.6,12.0))*(1-ease(seg(ph,16.0,16.9)));
    if(frost>0.01)ringS(ex,ey,rad*0.30,theme.red,(isDark?0.4:0.55)*frost,1.4*DPR,true);

    /* three gold ticks — K=3 */
    for(var i=0;i<3;i++){
      var pu=ease(seg(ph,12.2+i*0.7,12.8+i*0.7))*(1-ease(seg(ph,15.9,16.6)));
      if(pu<=0.01)continue;
      var tx2=ex+rad*0.30*Math.cos(-0.9+i*0.5),ty2=ey+rad*0.30*Math.sin(-0.9+i*0.5);
      if(isDark)gdot(tx2,ty2,rad*0.020,theme.gold,0.9*pu);
      else{disc(tx2,ty2,rad*0.015,theme.gold,0.85*pu);circle(tx2,ty2,rad*0.028,theme.gold,0.6*pu,1*DPR);}
    }

    /* alarm ring expands from the gripper */
    var al=seg(ph,14.6,15.6);
    if(al>0&&al<1){var au=ease(al);
      ringS(ex,ey,rad*(0.30+0.55*au),theme.gold,(1-au)*(isDark?0.65:0.85),1.6*DPR,false);}

    /* the fresh noise flies in and lands on the elbow */
    var fly=seg(ph,15.2,16.4);
    if(fly>0&&fly<1){var fu=ease(fly);
      var sx=bx-rad*1.5,sy=by-rad*1.55;
      var xx=sx+(J[0]-sx)*fu,yy=sy+(J[1]-sy)*fu-Math.sin(fu*Math.PI)*rad*0.30;
      if(isDark){gdot(xx,yy,rad*0.024,theme.gold,0.95);gdot(xx,yy,rad*0.06,theme.gold,0.25);}
      else{ctx.save();ctx.setLineDash([3*DPR,5*DPR]);ctx.beginPath();ctx.moveTo(sx,sy);
        ctx.quadraticCurveTo((sx+J[0])/2,Math.min(sy,yy)-rad*0.18,xx,yy);
        ctx.lineWidth=1.1*DPR;ctx.strokeStyle=rgba(theme.gold,0.5);ctx.stroke();ctx.setLineDash([]);ctx.restore();
        disc(xx,yy,rad*0.018,theme.gold,0.9);circle(xx,yy,rad*0.036,theme.gold,0.7,1.2*DPR);}
    }
    /* thaw pulse */
    var th=seg(ph,16.4,17.2);
    if(th>0&&th<1)ringS(J[0],J[1],rad*0.42*ease(th),theme.teal,(1-ease(th))*0.5,1.4*DPR,false);
  }

  function draw(ph){
    ctx.clearRect(0,0,W,H);ctx.save();ctx.translate(px.x,px.y);
    if(isDark){
      ctx.globalCompositeOperation='lighter';
      for(var s=0;s<stars.length;s++){var st=stars[s];var a=0.05+0.03*Math.sin(tSec*0.5+st.ph);gdot(MX(st.x*0.95+0.03),MY(st.y),st.r*DPR,[210,225,245],Math.max(0,a));}
    }else ctx.globalCompositeOperation='source-over';
    scene(ph);
    ctx.restore();
  }

  function frame(now){if(!running)return;if(!t0)t0=now;tSec=(now-t0)/1000;
    px.x+=(pxT.x-px.x)*0.06;px.y+=(pxT.y-px.y)*0.06;draw(tSec%P);raf=requestAnimationFrame(frame);}

  function resize(){DPR=Math.min(2,window.devicePixelRatio||1);var rect=canvas.getBoundingClientRect();W=Math.max(1,Math.round(rect.width*DPR));H=Math.max(1,Math.round(rect.height*DPR));canvas.width=W;canvas.height=H;if(!running)draw((tSec||4)%P);}
  function start(){if(running||reduce)return;running=true;t0=0;raf=requestAnimationFrame(frame);}
  function stop(){running=false;cancelAnimationFrame(raf);}

  window.radHero={
    setTheme:function(name){isDark=(name!=='light');theme=isDark?THEMES.dark:THEMES.light;if(!running||reduce)draw((reduce?4:tSec)%P);},
    seek:function(t){stop();tSec=t;draw(t%P);},
    pause:stop,resume:start
  };

  isDark=document.documentElement.dataset.theme!=='light';
  theme=isDark?THEMES.dark:THEMES.light;
  resize();
  var rt2;window.addEventListener('resize',function(){clearTimeout(rt2);rt2=setTimeout(resize,150);},{passive:true});
  if(matchMedia('(pointer:fine)').matches){window.addEventListener('pointermove',function(e){var mx=(e.clientX/window.innerWidth-0.5),my=(e.clientY/window.innerHeight-0.5);pxT.x=mx*(isDark?16:9)*DPR;pxT.y=my*(isDark?12:7)*DPR;},{passive:true});}
  document.addEventListener('visibilitychange',function(){document.hidden?stop():start();});
  if('IntersectionObserver' in window){new IntersectionObserver(function(es){es.forEach(function(en){en.isIntersecting?start():stop();});},{threshold:0.04}).observe(canvas);}else{start();}
  if(reduce)draw(4);else start();
})();
