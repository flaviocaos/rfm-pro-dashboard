import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";

const SEGS = {
  VIP:        { hex:"#8b5cf6", bg:"#f5f3ff", tx:"#4c1d95", emoji:"👑", desc:"Alto valor, alta freq., recente" },
  Leal:       { hex:"#10b981", bg:"#ecfdf5", tx:"#064e3b", emoji:"💎", desc:"Frequência média-alta, bom valor" },
  "Em Risco": { hex:"#f59e0b", bg:"#fffbeb", tx:"#78350f", emoji:"⚠️", desc:"Foram bons, recência caindo" },
  Inativo:    { hex:"#ef4444", bg:"#fef2f2", tx:"#7f1d1d", emoji:"💤", desc:"Baixa freq., baixo valor" },
  Novo:       { hex:"#3b82f6", bg:"#eff6ff", tx:"#1e3a8a", emoji:"🌱", desc:"Poucas compras, potencial alto" },
};
const SK = Object.keys(SEGS);
const CHURN_MAP = {"#8b5cf6":0.05,"#10b981":0.15,"#f59e0b":0.55,"#ef4444":0.80,"#3b82f6":0.30};

const fR = v => `R$ ${(v||0).toLocaleString("pt-BR",{minimumFractionDigits:2,maximumFractionDigits:2})}`;
const fN = v => (v||0).toLocaleString("pt-BR");
const fK = v => v>=1000000?`${(v/1000000).toFixed(1)}M`:v>=1000?`${(v/1000).toFixed(1)}k`:String(Math.round(v||0));

// ── TensorFlow.js ──
let tfModel = null, scalerParams = null;
async function loadModel() {
  try {
    scalerParams = await fetch("/scaler_params.json").then(r=>r.json());
    tfModel = await tf.loadLayersModel("/model/model.json");
    return true;
  } catch(e) { return false; }
}
function scalerTransform(rec,freq,mon) {
  if(!scalerParams) return [rec,freq,mon];
  const [mr,mf,mm]=scalerParams.mean,[sr,sf,sm]=scalerParams.scale;
  return [(rec-mr)/sr,(freq-mf)/sf,(mon-mm)/sm];
}
async function predictSegmentTF(rec,freq,mon) {
  if(!tfModel||!scalerParams) return null;
  const scaled=scalerTransform(rec,freq,mon);
  const input=tf.tensor2d([scaled]);
  const pred=tfModel.predict(input);
  const clusterIdx=(await pred.argMax(1).data())[0];
  const probs=await pred.data();
  input.dispose();pred.dispose();
  const sc=v=>v>=.75?5:v>=.5?4:v>=.35?3:v>=.2?2:1;
  const recMax=scalerParams.mean[0]+scalerParams.scale[0]*2;
  const rn=Math.max(0,Math.min(1,1-rec/recMax));
  const fn=Math.max(0,Math.min(1,freq/20));
  const mn=Math.max(0,Math.min(1,mon/15000));
  const rs=sc(rn),fs=sc(fn),ms=sc(mn);
  const seg=rs>=4&&fs>=4&&ms>=4?"VIP":(rs+fs+ms)/3>=3.5?"Leal":rs<=2&&(fs>=3||ms>=3)?"Em Risco":fs<=2&&ms<=2?"Inativo":"Novo";
  return{seg,probs:Array.from(probs),clusterIdx,rs,fs,ms};
}

// ── Data ──
function genRows(n=180){
  const now=new Date("2024-12-31"),out=[];
  const profiles=[
    {recRange:[1,30],freqRange:[8,20],valRange:[2000,15000],weight:15},
    {recRange:[1,90],freqRange:[4,12],valRange:[800,5000],weight:25},
    {recRange:[90,200],freqRange:[3,8],valRange:[500,3000],weight:20},
    {recRange:[200,365],freqRange:[1,3],valRange:[50,500],weight:25},
    {recRange:[1,60],freqRange:[1,3],valRange:[100,1000],weight:15},
  ];
  const totalW=profiles.reduce((a,p)=>a+p.weight,0);
  for(let i=1;i<=n;i++){
    let rnd=Math.random()*totalW,prof=profiles[0];
    for(const p of profiles){rnd-=p.weight;if(rnd<=0){prof=p;break;}}
    const purchases=Math.floor(Math.random()*(prof.freqRange[1]-prof.freqRange[0]))+prof.freqRange[0];
    for(let j=0;j<purchases;j++){
      const d=Math.floor(Math.random()*(prof.recRange[1]-prof.recRange[0]))+prof.recRange[0];
      out.push({customer_id:`C${String(i).padStart(4,"0")}`,purchase_date:new Date(now-(d+Math.floor(Math.random()*60))*864e5).toISOString().slice(0,10),purchase_value:parseFloat((Math.random()*(prof.valRange[1]-prof.valRange[0])+prof.valRange[0]).toFixed(2))});
    }
  }
  return out;
}

function buildClients(rows,k=5){
  const map={},now=new Date("2024-12-31");
  rows.forEach(r=>{
    const id=r.customer_id,d=new Date(r.purchase_date);
    if(!map[id])map[id]={id,last:d,freq:0,mon:0};
    if(d>map[id].last)map[id].last=d;
    map[id].freq++;map[id].mon+=r.purchase_value;
  });
  let arr=Object.values(map).map(c=>({...c,rec:Math.floor((now-c.last)/864e5),mon:parseFloat(c.mon.toFixed(2))}));
  const nm=(a,key)=>{const vs=a.map(x=>x[key]),mn=Math.min(...vs),mx=Math.max(...vs);return a.map(x=>({...x,[key+"_n"]:mx===mn?.5:(x[key]-mn)/(mx-mn)}));};
  arr=nm(arr,"rec");arr=nm(arr,"freq");arr=nm(arr,"mon");
  arr=arr.map(c=>({...c,rn:1-c.rec_n,fn:c.freq_n,mn2:c.mon_n}));
  arr=nm(arr,"rn");arr=nm(arr,"fn");arr=nm(arr,"mn2");
  let cents=arr.slice(0,k).map(d=>({r:d.rn_n,f:d.fn_n,m:d.mn2_n})),asgn=arr.map(()=>0);
  for(let it=0;it<50;it++){
    asgn=arr.map(d=>{let b=0,bd=Infinity;cents.forEach((c,ci)=>{const dist=(d.rn_n-c.r)**2+(d.fn_n-c.f)**2+(d.mn2_n-c.m)**2;if(dist<bd){bd=dist;b=ci;}});return b;});
    cents=Array.from({length:k},(_,ci)=>{const pts=arr.filter((_,i)=>asgn[i]===ci);if(!pts.length)return cents[ci];return{r:pts.reduce((s,p)=>s+p.rn_n,0)/pts.length,f:pts.reduce((s,p)=>s+p.fn_n,0)/pts.length,m:pts.reduce((s,p)=>s+p.mn2_n,0)/pts.length};});
  }
  const sc=v=>v>=.75?5:v>=.5?4:v>=.35?3:v>=.2?2:1;
  const sg=(r,f,m)=>r>=4&&f>=4&&m>=4?"VIP":(r+f+m)/3>=3.5?"Leal":r<=2&&(f>=3||m>=3)?"Em Risco":f<=2&&m<=2?"Inativo":"Novo";
  return arr.map((c,i)=>{
    const rs=sc(c.rn_n),fs=sc(c.fn_n),ms=sc(c.mn2_n),seg=sg(rs,fs,ms);
    const base=CHURN_MAP[SEGS[seg].hex]||0.3;
    return{...c,cluster:asgn[i],rs,fs,ms,segment:seg,churn:Math.min(1,Math.max(0,base+(Math.random()-.5)*.08)).toFixed(2),rn_n:c.rn_n,fn_n:c.fn_n,mn2_n:c.mn2_n};
  });
}

function parseCSV(txt){
  const lines=txt.trim().split("\n"),h=lines[0].split(",").map(x=>x.trim().toLowerCase().replace(/\s+/g,"_").replace(/['"]/g,""));
  return lines.slice(1).map(l=>{const v=l.split(","),o={};h.forEach((k,i)=>o[k]=(v[i]||"").trim().replace(/['"]/g,""));o.purchase_value=parseFloat(o.purchase_value)||0;return o;}).filter(r=>r.customer_id&&r.purchase_date&&r.purchase_value>0);
}

function exportCSV(clients,seg){
  const data=seg==="Todos"?clients:clients.filter(c=>c.segment===seg);
  const hdr="customer_id,segment,recency,frequency,monetary,r_score,f_score,m_score";
  const rows=data.map(c=>`${c.id},${c.segment},${c.rec},${c.freq},${c.mon.toFixed(2)},${c.rs},${c.fs},${c.ms}`);
  const blob=new Blob([[hdr,...rows].join("\n")],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`rfm_${seg}.csv`;a.click();
}

// ── Charts ──
function HistogramCanvas({data,keyName,label,color,bins=15}){
  const ref=useRef();
  useEffect(()=>{
    const cv=ref.current;if(!cv||!data.length)return;
    const ctx=cv.getContext("2d"),W=cv.width,H=cv.height,P={t:20,r:16,b:36,l:48};
    ctx.clearRect(0,0,W,H);
    const vals=data.map(d=>d[keyName]);
    const mn=Math.min(...vals),mx=Math.max(...vals),range=mx-mn||1;
    const bw=range/bins,counts=new Array(bins).fill(0);
    vals.forEach(v=>{const bi=Math.min(Math.floor((v-mn)/bw),bins-1);counts[bi]++;});
    const maxC=Math.max(...counts,1);
    const cW=(W-P.l-P.r)/bins,cH=H-P.t-P.b;
    ctx.strokeStyle="rgba(99,102,241,0.08)";ctx.lineWidth=1;
    for(let i=0;i<=4;i++){
      const y=P.t+cH-(cH/4)*i;
      ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();
      ctx.fillStyle="#94a3b8";ctx.font="10px system-ui";ctx.textAlign="right";
      ctx.fillText(Math.round(maxC/4*i),P.l-4,y+4);
    }
    counts.forEach((c,i)=>{
      const x=P.l+i*cW,bH=(c/maxC)*cH,y=P.t+cH-bH;
      const grad=ctx.createLinearGradient(0,y,0,y+bH);
      grad.addColorStop(0,color);grad.addColorStop(1,color+"55");
      ctx.fillStyle=grad;ctx.beginPath();ctx.roundRect(x+1,y,cW-2,bH,2);ctx.fill();
    });
    ctx.fillStyle="#94a3b8";ctx.font="10px system-ui";ctx.textAlign="center";
    [0,.25,.5,.75,1].forEach(t=>{
      const v=mn+t*range,x=P.l+t*(W-P.l-P.r);
      ctx.fillText(v>=1000?`${(v/1000).toFixed(1)}k`:Math.round(v),x,H-6);
    });
    ctx.fillStyle="#334155";ctx.font="bold 11px system-ui";ctx.textAlign="center";
    ctx.fillText(label,W/2,13);
  },[data,keyName]);
  return <canvas ref={ref} width={280} height={180} style={{width:"100%",height:"auto",display:"block"}}/>;
}

function ElbowCanvas({clients}){
  const ref=useRef();
  useEffect(()=>{
    const cv=ref.current;if(!cv||!clients.length)return;
    const ctx=cv.getContext("2d"),W=cv.width,H=cv.height,P={t:28,r:24,b:40,l:52};
    ctx.clearRect(0,0,W,H);
    const pts=clients.map(c=>[c.rn_n||0,c.fn_n||0,c.mn2_n||0]);
    const ks=[2,3,4,5,6,7,8];
    const inertias=ks.map(k=>{
      const sample=pts.slice(0,Math.min(pts.length,120));
      let cents=sample.slice(0,k).map(p=>[...p]);
      let labels=new Array(sample.length).fill(0);
      for(let it=0;it<15;it++){
        labels=sample.map(p=>{let b=0,bd=Infinity;cents.forEach((c,ci)=>{const d=p.reduce((s,v,i)=>s+(v-c[i])**2,0);if(d<bd){bd=d;b=ci;}});return b;});
        cents=Array.from({length:k},(_,ci)=>{const ps=sample.filter((_,i)=>labels[i]===ci);if(!ps.length)return cents[ci];return ps[0].map((_,j)=>ps.reduce((s,p)=>s+p[j],0)/ps.length);});
      }
      return sample.reduce((s,p,i)=>s+p.reduce((ss,v,j)=>ss+(v-cents[labels[i]][j])**2,0),0);
    });
    const maxI=Math.max(...inertias),minI=Math.min(...inertias),range=maxI-minI||1;
    const cH=H-P.t-P.b,xW=(W-P.l-P.r)/(ks.length-1);
    ctx.strokeStyle="rgba(99,102,241,0.08)";ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const y=P.t+cH/4*i;ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();}
    // Area
    ctx.beginPath();
    inertias.forEach((v,i)=>{const x=P.l+i*xW,y=P.t+cH*(1-(v-minI)/range);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
    ctx.lineTo(P.l+(ks.length-1)*xW,P.t+cH);ctx.lineTo(P.l,P.t+cH);ctx.closePath();
    const grad=ctx.createLinearGradient(0,P.t,0,P.t+cH);
    grad.addColorStop(0,"rgba(99,102,241,0.25)");grad.addColorStop(1,"rgba(99,102,241,0.02)");
    ctx.fillStyle=grad;ctx.fill();
    // Line
    ctx.beginPath();ctx.strokeStyle="#6366f1";ctx.lineWidth=2.5;ctx.lineJoin="round";
    inertias.forEach((v,i)=>{const x=P.l+i*xW,y=P.t+cH*(1-(v-minI)/range);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
    ctx.stroke();
    // Points
    inertias.forEach((v,i)=>{
      const x=P.l+i*xW,y=P.t+cH*(1-(v-minI)/range);
      ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);
      ctx.fillStyle=i===3?"#f59e0b":"#6366f1";ctx.fill();
      ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.stroke();
      ctx.fillStyle="#475569";ctx.font="10px system-ui";ctx.textAlign="center";
      ctx.fillText(`k=${ks[i]}`,x,H-8);
    });
    // Elbow
    const ex=P.l+3*xW,ey=P.t+cH*(1-(inertias[3]-minI)/range);
    ctx.strokeStyle="#f59e0b44";ctx.lineWidth=1;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(ex,ey-10);ctx.lineTo(ex,P.t+cH);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle="#f59e0b";ctx.font="bold 9px system-ui";ctx.textAlign="center";ctx.fillText("⬆ Cotovelo",ex,ey-14);
    ctx.fillStyle="#1e293b";ctx.font="bold 12px system-ui";ctx.textAlign="center";ctx.fillText("Método do Cotovelo — Inércia por k",W/2,16);
    ctx.save();ctx.translate(14,H/2);ctx.rotate(-Math.PI/2);ctx.fillStyle="#94a3b8";ctx.font="10px system-ui";ctx.textAlign="center";ctx.fillText("Inércia",0,0);ctx.restore();
  },[clients]);
  return <canvas ref={ref} width={480} height={220} style={{width:"100%",height:"auto",display:"block"}}/>;
}

function LatentCanvas({clients}){
  const ref=useRef();
  useEffect(()=>{
    const cv=ref.current;if(!cv||!clients.length)return;
    const ctx=cv.getContext("2d"),W=cv.width,H=cv.height,P=44;
    ctx.clearRect(0,0,W,H);
    ctx.strokeStyle="rgba(99,102,241,0.06)";ctx.lineWidth=1;
    for(let i=0;i<=5;i++){
      ctx.beginPath();ctx.moveTo(P+i*(W-P*2)/5,P*.5);ctx.lineTo(P+i*(W-P*2)/5,H-P);ctx.stroke();
      ctx.beginPath();ctx.moveTo(P,P*.5+i*(H-P*1.5)/5);ctx.lineTo(W-P,P*.5+i*(H-P*1.5)/5);ctx.stroke();
    }
    clients.forEach(c=>{
      const x=P+(c.rn_n||0)*(W-P*2),y=H-P-(c.mn2_n||0)*(H-P*1.5);
      ctx.beginPath();ctx.arc(x,y,3.5,0,Math.PI*2);
      ctx.fillStyle=SEGS[c.segment].hex+"99";ctx.fill();
    });
    SK.forEach(sg=>{
      const spts=clients.filter(c=>c.segment===sg);if(!spts.length)return;
      const cx=P+spts.reduce((s,c)=>s+(c.rn_n||0),0)/spts.length*(W-P*2);
      const cy=H-P-spts.reduce((s,c)=>s+(c.mn2_n||0),0)/spts.length*(H-P*1.5);
      ctx.beginPath();ctx.arc(cx,cy,14,0,Math.PI*2);
      ctx.fillStyle=SEGS[sg].hex+"22";ctx.fill();
      ctx.strokeStyle=SEGS[sg].hex;ctx.lineWidth=2;ctx.stroke();
      ctx.fillStyle=SEGS[sg].hex;ctx.font="bold 11px system-ui";ctx.textAlign="center";ctx.fillText(SEGS[sg].emoji,cx,cy+4);
    });
    ctx.fillStyle="#94a3b8";ctx.font="10px system-ui";ctx.textAlign="center";ctx.fillText("← baixa   Dim. 1 — Recência Norm.   alta →",W/2,H-6);
    ctx.save();ctx.translate(12,H/2);ctx.rotate(-Math.PI/2);ctx.fillText("Dim. 2 — Monetário",0,0);ctx.restore();
    ctx.fillStyle="#1e293b";ctx.font="bold 12px system-ui";ctx.textAlign="center";ctx.fillText("Espaço Latente 2D — Projeção dos Clusters",W/2,14);
  },[clients]);
  return <canvas ref={ref} width={560} height={280} style={{width:"100%",height:"auto",display:"block"}}/>;
}

function ClusterProfileCanvas({clients}){
  const ref=useRef();
  useEffect(()=>{
    const cv=ref.current;if(!cv||!clients.length)return;
    const ctx=cv.getContext("2d"),W=cv.width,H=cv.height,P={t:30,r:20,b:50,l:50};
    ctx.clearRect(0,0,W,H);
    const segData=SK.map(sg=>{
      const pts=clients.filter(c=>c.segment===sg);if(!pts.length)return{seg:sg,values:[0,0,0]};
      return{seg:sg,values:[pts.reduce((a,c)=>a+c.rec,0)/pts.length,pts.reduce((a,c)=>a+c.freq,0)/pts.length,pts.reduce((a,c)=>a+c.mon,0)/pts.length]};
    });
    const maxVals=[Math.max(...segData.map(s=>s.values[0]),1),Math.max(...segData.map(s=>s.values[1]),1),Math.max(...segData.map(s=>s.values[2]),1)];
    const normData=segData.map(s=>({...s,norm:s.values.map((v,i)=>v/maxVals[i])}));
    const labels=["Recência","Frequência","Monetário"];
    const groupW=(W-P.l-P.r)/3;
    const barW=groupW/(SK.length+1);
    ctx.strokeStyle="rgba(99,102,241,0.08)";ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const y=P.t+(H-P.t-P.b)/4*(4-i);ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();ctx.fillStyle="#94a3b8";ctx.font="9px system-ui";ctx.textAlign="right";ctx.fillText(`${i*25}%`,P.l-4,y+3);}
    labels.forEach((lbl,gi)=>{
      normData.forEach((s,si)=>{
        const x=P.l+gi*groupW+si*barW+barW/2,bH=s.norm[gi]*(H-P.t-P.b),y=P.t+(H-P.t-P.b)-bH;
        const grad=ctx.createLinearGradient(0,y,0,y+bH);
        grad.addColorStop(0,SEGS[s.seg].hex);grad.addColorStop(1,SEGS[s.seg].hex+"44");
        ctx.fillStyle=grad;ctx.beginPath();ctx.roundRect(x,y,barW-2,bH,3);ctx.fill();
      });
      ctx.fillStyle="#475569";ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(lbl,P.l+gi*groupW+groupW/2,H-P.b+14);
      if(gi<2){ctx.strokeStyle="rgba(99,102,241,0.1)";ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(P.l+(gi+1)*groupW,P.t);ctx.lineTo(P.l+(gi+1)*groupW,H-P.b);ctx.stroke();ctx.setLineDash([]);}
    });
    ctx.fillStyle="#1e293b";ctx.font="bold 12px system-ui";ctx.textAlign="center";ctx.fillText("Perfis por Segmento — Médias Normalizadas",W/2,16);
    SK.forEach((sg,i)=>{
      const x=P.l+i*(W-P.l-P.r)/SK.length+8;
      ctx.fillStyle=SEGS[sg].hex;ctx.fillRect(x,H-10,10,6);
      ctx.fillStyle="#64748b";ctx.font="9px system-ui";ctx.textAlign="left";ctx.fillText(sg,x+13,H-5);
    });
  },[clients]);
  return <canvas ref={ref} width={680} height={260} style={{width:"100%",height:"auto",display:"block"}}/>;
}

function RFMHeatmapCanvas({clients}){
  const ref=useRef();
  useEffect(()=>{
    const cv=ref.current;if(!cv||!clients.length)return;
    const ctx=cv.getContext("2d"),W=cv.width,H=cv.height,P={t:30,r:20,b:30,l:60};
    ctx.clearRect(0,0,W,H);
    const cellW=(W-P.l-P.r)/5,cellH=(H-P.t-P.b)/5;
    const matrix=Array.from({length:5},()=>new Array(5).fill(0));
    clients.forEach(c=>{const ri=(c.rs||1)-1,fi=(c.fs||1)-1;if(ri>=0&&ri<5&&fi>=0&&fi<5)matrix[4-ri][fi]++;});
    const maxV=Math.max(...matrix.flat(),1);
    matrix.forEach((row,ri)=>{
      row.forEach((v,fi)=>{
        const x=P.l+fi*cellW,y=P.t+ri*cellH,intensity=v/maxV;
        ctx.fillStyle=`rgba(${Math.round(59+136*intensity)},${Math.round(130-80*intensity)},${Math.round(246-180*intensity)},${0.08+intensity*0.85})`;
        ctx.fillRect(x+1,y+1,cellW-2,cellH-2);
        if(v>0){ctx.fillStyle=intensity>0.5?"#fff":"#1e293b";ctx.font=`bold ${Math.min(16,9+v)}px system-ui`;ctx.textAlign="center";ctx.fillText(v,x+cellW/2,y+cellH/2+5);}
      });
    });
    for(let i=1;i<=5;i++){
      ctx.fillStyle="#64748b";ctx.font="10px system-ui";ctx.textAlign="center";ctx.fillText(`F=${i}`,P.l+(i-1)*cellW+cellW/2,P.t-8);
      ctx.textAlign="right";ctx.fillText(`R=${6-i}`,P.l-6,P.t+(i-1)*cellH+cellH/2+4);
    }
    ctx.fillStyle="#1e293b";ctx.font="bold 12px system-ui";ctx.textAlign="center";ctx.fillText("Matriz RFM — Recência × Frequência",W/2,16);
  },[clients]);
  return <canvas ref={ref} width={480} height={280} style={{width:"100%",height:"auto",display:"block"}}/>;
}

function ChurnBarCanvas({clients}){
  const ref=useRef();
  useEffect(()=>{
    const cv=ref.current;if(!cv)return;
    const ctx=cv.getContext("2d"),W=cv.width,H=cv.height,P={t:28,r:20,b:44,l:52};
    ctx.clearRect(0,0,W,H);
    const barW=(W-P.l-P.r)/(SK.length*1.6),gap=(W-P.l-P.r)/SK.length-barW;
    ctx.strokeStyle="rgba(99,102,241,0.08)";ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const y=P.t+(H-P.t-P.b)/4*i;ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();ctx.fillStyle="#94a3b8";ctx.font="9px system-ui";ctx.textAlign="right";ctx.fillText(`${100-i*25}%`,P.l-4,y+3);}
    SK.forEach((sg,i)=>{
      const churn=CHURN_MAP[SEGS[sg].hex]||0.3;
      const x=P.l+i*(barW+gap)+gap/2,bH=churn*(H-P.t-P.b),y=P.t+(1-churn)*(H-P.t-P.b);
      const grad=ctx.createLinearGradient(0,y,0,y+bH);
      grad.addColorStop(0,SEGS[sg].hex);grad.addColorStop(1,SEGS[sg].hex+"44");
      ctx.fillStyle=grad;ctx.beginPath();ctx.roundRect(x,y,barW,bH,4);ctx.fill();
      ctx.fillStyle=SEGS[sg].hex;ctx.font="bold 11px system-ui";ctx.textAlign="center";ctx.fillText(`${Math.round(churn*100)}%`,x+barW/2,y-5);
      ctx.fillStyle="#475569";ctx.font="10px system-ui";ctx.fillText(SEGS[sg].emoji,x+barW/2,H-26);
      ctx.fillText(sg,x+barW/2,H-12);
    });
    ctx.fillStyle="#1e293b";ctx.font="bold 12px system-ui";ctx.textAlign="center";ctx.fillText("Risco de Churn por Segmento",W/2,16);
  },[clients]);
  return <canvas ref={ref} width={480} height={220} style={{width:"100%",height:"auto",display:"block"}}/>;
}

function ScatterCanvas({data,filter,hovId,setHov}){
  const ref=useRef();
  useEffect(()=>{
    const cv=ref.current;if(!cv)return;
    const ctx=cv.getContext("2d"),W=cv.width,H=cv.height,P=48;
    ctx.clearRect(0,0,W,H);
    const pts=filter==="Todos"?data:data.filter(c=>c.segment===filter);
    if(!pts.length)return;
    const xs=pts.map(c=>c.rec),ys=pts.map(c=>c.mon),zM=Math.max(...pts.map(c=>c.freq));
    const xn=Math.min(...xs),xx=Math.max(...xs),yn=Math.min(...ys),yx=Math.max(...ys);
    const tx=v=>P+(v-xn)/(xx-xn||1)*(W-P*2),ty=v=>H-P-(v-yn)/(yx-yn||1)*(H-P*1.7);
    ctx.strokeStyle="rgba(99,102,241,0.06)";ctx.lineWidth=1;
    for(let i=0;i<=5;i++){ctx.beginPath();ctx.moveTo(P+i*(W-P*2)/5,P*.4);ctx.lineTo(P+i*(W-P*2)/5,H-P);ctx.stroke();ctx.beginPath();ctx.moveTo(P,P*.4+i*(H-P*1.7)/5);ctx.lineTo(W-P,P*.4+i*(H-P*1.7)/5);ctx.stroke();}
    ctx.fillStyle="rgba(100,116,139,0.4)";ctx.font="11px system-ui";ctx.textAlign="center";
    ctx.fillText("← mais recente   Recência (dias)   mais antigo →",W/2,H-8);
    ctx.save();ctx.translate(15,H/2);ctx.rotate(-Math.PI/2);ctx.fillText("Valor total",0,0);ctx.restore();
    pts.forEach(c=>{
      const x=tx(c.rec),y=ty(c.mon),r=3+c.freq/zM*12,hov=c.id===hovId;
      ctx.beginPath();ctx.arc(x,y,hov?r+4:r,0,Math.PI*2);
      if(hov){ctx.shadowBlur=16;ctx.shadowColor=SEGS[c.segment].hex;}
      ctx.fillStyle=SEGS[c.segment].hex+(hov?"ff":"90");ctx.fill();ctx.shadowBlur=0;
      if(hov){ctx.strokeStyle="#fff";ctx.lineWidth=2.5;ctx.stroke();}
    });
  },[data,filter,hovId]);
  const onMove=e=>{
    const cv=ref.current;if(!cv)return;
    const rect=cv.getBoundingClientRect(),mx=(e.clientX-rect.left)*(cv.width/rect.width),my=(e.clientY-rect.top)*(cv.height/rect.height);
    const pts=filter==="Todos"?data:data.filter(c=>c.segment===filter);
    const P=48,W=cv.width,H=cv.height;
    const xs=pts.map(c=>c.rec),ys=pts.map(c=>c.mon);
    const xn=Math.min(...xs),xx=Math.max(...xs),yn=Math.min(...ys),yx=Math.max(...ys);
    const tx=v=>P+(v-xn)/(xx-xn||1)*(W-P*2),ty=v=>H-P-(v-yn)/(yx-yn||1)*(H-P*1.7);
    let found=null,best=Infinity;
    pts.forEach(c=>{const dx=tx(c.rec)-mx,dy=ty(c.mon)-my,d=dx*dx+dy*dy;if(d<best&&d<600){best=d;found=c;}});
    setHov(found||null);
  };
  return <canvas ref={ref} width={700} height={300} style={{width:"100%",height:"auto",cursor:"crosshair",display:"block"}} onMouseMove={onMove} onMouseLeave={()=>setHov(null)}/>;
}

// ── Small components ──
function MiniBar({value,max,color}){return <div style={{flex:1,height:6,borderRadius:3,background:"#f1f5f9",overflow:"hidden"}}><div style={{width:`${Math.min((value/max)*100,100)}%`,height:"100%",background:color,borderRadius:3,transition:"width .3s"}}/></div>;}
function ChurnRing({value}){const r=20,circ=2*Math.PI*r,pct=Math.min(parseFloat(value)||0,1);const col=pct>.6?"#ef4444":pct>.35?"#f59e0b":"#10b981";return <svg width={56} height={56} viewBox="0 0 56 56"><circle cx={28} cy={28} r={r} fill="none" stroke="#f1f5f9" strokeWidth={5}/><circle cx={28} cy={28} r={r} fill="none" stroke={col} strokeWidth={5} strokeDasharray={circ} strokeDashoffset={circ*(1-pct)} strokeLinecap="round" transform="rotate(-90 28 28)"/><text x={28} y={33} textAnchor="middle" fontSize={11} fontWeight={600} fill={col}>{Math.round(pct*100)}%</text></svg>;}
function ScoreBar({v}){return <div style={{display:"flex",gap:2}}>{[1,2,3,4,5].map(i=><div key={i} style={{width:12,height:4,borderRadius:2,background:i<=v?(v>=4?"#10b981":v>=3?"#f59e0b":"#ef4444"):"#e2e8f0"}}/>)}</div>;}

// ── Analytics Page ──
function AnalyticsPage({clients,filter}){
  const card={background:"#fff",borderRadius:14,border:"1px solid #e2e8f0",padding:"16px 18px"};
  const filtered=filter==="Todos"?clients:clients.filter(c=>c.segment===filter);
  const [hovS,setHovS]=useState(null);
  return(
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      <div style={{background:"linear-gradient(135deg,#1e1b4b,#312e81)",borderRadius:14,padding:"16px 20px",color:"#fff",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{fontSize:24}}>📊</div>
          <div><div style={{fontSize:14,fontWeight:700}}>Análise Completa — {filtered.length} clientes</div><div style={{fontSize:11,color:"rgba(255,255,255,0.6)"}}>Gráficos gerados automaticamente com os dados carregados · Filtre por segmento no header</div></div>
        </div>
        <div style={{fontSize:11,color:"rgba(255,255,255,0.5)"}}>📁 Suba qualquer CSV para atualizar</div>
      </div>

      {/* Histogramas */}
      <div style={card}>
        <div style={{fontSize:13,fontWeight:700,marginBottom:12,color:"#1e293b"}}>📈 Distribuição RFM — Histogramas</div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12}}>
          {[["rec","Recência (dias)","#6366f1"],["freq","Frequência (compras)","#10b981"],["mon","Monetário (R$)","#f59e0b"]].map(([k,l,c])=>(
            <div key={k} style={{background:"#f8fafc",borderRadius:10,padding:12}}>
              <HistogramCanvas data={filtered} keyName={k} label={l} color={c}/>
            </div>
          ))}
        </div>
      </div>

      {/* Elbow + Latent */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <div style={card}>
          <div style={{fontSize:13,fontWeight:700,marginBottom:10}}>🔵 Método do Cotovelo (Elbow)</div>
          <div style={{background:"#f8fafc",borderRadius:10,padding:12}}><ElbowCanvas clients={filtered}/></div>
          <div style={{fontSize:11,color:"#94a3b8",marginTop:8}}>Mostra o k ideal de clusters — o "cotovelo" indica onde adicionar mais clusters não traz ganho significativo.</div>
        </div>
        <div style={card}>
          <div style={{fontSize:13,fontWeight:700,marginBottom:10}}>🌐 Espaço Latente 2D</div>
          <div style={{background:"#f8fafc",borderRadius:10,padding:12}}><LatentCanvas clients={filtered}/></div>
          <div style={{fontSize:11,color:"#94a3b8",marginTop:8}}>Projeção 2D dos clusters no espaço normalizado. Centróides marcados com emoji do segmento.</div>
        </div>
      </div>

      {/* Cluster Profiles */}
      <div style={card}>
        <div style={{fontSize:13,fontWeight:700,marginBottom:10}}>🎯 Perfis por Segmento — Comparação de Médias</div>
        <div style={{background:"#f8fafc",borderRadius:10,padding:12}}><ClusterProfileCanvas clients={filtered}/></div>
        <div style={{display:"flex",gap:16,flexWrap:"wrap",marginTop:10}}>
          {SK.map(sg=><span key={sg} style={{fontSize:11,display:"flex",alignItems:"center",gap:5,color:"#64748b"}}><span style={{width:10,height:10,borderRadius:2,background:SEGS[sg].hex,display:"inline-block"}}/>{SEGS[sg].emoji} {sg}</span>)}
        </div>
      </div>

      {/* Heatmap + Churn */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <div style={card}>
          <div style={{fontSize:13,fontWeight:700,marginBottom:10}}>🔥 Matriz RFM — Heatmap</div>
          <div style={{background:"#f8fafc",borderRadius:10,padding:12}}><RFMHeatmapCanvas clients={filtered}/></div>
          <div style={{fontSize:11,color:"#94a3b8",marginTop:8}}>Concentração de clientes por score R×F. Células mais escuras = mais clientes.</div>
        </div>
        <div style={card}>
          <div style={{fontSize:13,fontWeight:700,marginBottom:10}}>⚠️ Risco de Churn por Segmento</div>
          <div style={{background:"#f8fafc",borderRadius:10,padding:12}}><ChurnBarCanvas clients={filtered}/></div>
          <div style={{fontSize:11,color:"#94a3b8",marginTop:8}}>Probabilidade estimada de churn por segmento baseada nos padrões RFM.</div>
        </div>
      </div>

      {/* Scatter interativo */}
      <div style={card}>
        <div style={{fontSize:13,fontWeight:700,marginBottom:6}}>🔵 Dispersão RFM Interativa</div>
        <div style={{fontSize:11,color:"#94a3b8",marginBottom:10}}>Passe o mouse sobre os pontos para ver detalhes · X = Recência · Y = Monetário · Tamanho = Frequência</div>
        <div style={{background:"#f8fafc",borderRadius:10,padding:12}}>
          <ScatterCanvas data={filtered} filter={filter} hovId={hovS?.id} setHov={setHovS}/>
        </div>
        {hovS&&<div style={{marginTop:10,padding:"10px 14px",borderRadius:8,background:SEGS[hovS.segment].bg,border:`1px solid ${SEGS[hovS.segment].hex}44`,display:"flex",gap:16,flexWrap:"wrap",alignItems:"center"}}>
          <span style={{fontFamily:"monospace",fontSize:12,fontWeight:700}}>{hovS.id}</span>
          <span style={{padding:"2px 8px",borderRadius:12,fontSize:11,background:SEGS[hovS.segment].hex+"22",color:SEGS[hovS.segment].tx,fontWeight:600}}>{SEGS[hovS.segment].emoji} {hovS.segment}</span>
          {[["Recência",`${hovS.rec}d`],["Frequência",`${hovS.freq}x`],["Monetário",fK(hovS.mon)]].map(([l,v])=><span key={l} style={{fontSize:11,color:"#64748b"}}><strong>{l}:</strong> {v}</span>)}
        </div>}
        <div style={{display:"flex",gap:14,flexWrap:"wrap",marginTop:10}}>
          {SK.map(sg=><span key={sg} style={{fontSize:11,display:"flex",alignItems:"center",gap:4,color:"#64748b"}}><span style={{width:8,height:8,borderRadius:"50%",background:SEGS[sg].hex,display:"inline-block"}}/>{sg}</span>)}
        </div>
      </div>

      {/* Stats table */}
      <div style={card}>
        <div style={{fontSize:13,fontWeight:700,marginBottom:12}}>📋 Estatísticas Descritivas</div>
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
            <thead><tr style={{borderBottom:"2px solid #f1f5f9"}}>{["Métrica","N","Mín","Máx","Média","Mediana","Desvio Padrão"].map(h=><th key={h} style={{padding:"8px 12px",textAlign:"left",fontSize:11,fontWeight:600,color:"#64748b"}}>{h}</th>)}</tr></thead>
            <tbody>
              {[["Recência (dias)","rec"],["Frequência (compras)","freq"],["Monetário (R$)","mon"]].map(([label,key],ri)=>{
                const vals=filtered.map(c=>c[key]).filter(v=>v!=null).sort((a,b)=>a-b);
                if(!vals.length)return null;
                const mean=vals.reduce((a,b)=>a+b,0)/vals.length;
                const median=vals.length%2===0?(vals[vals.length/2-1]+vals[vals.length/2])/2:vals[Math.floor(vals.length/2)];
                const std=Math.sqrt(vals.reduce((s,v)=>s+(v-mean)**2,0)/vals.length);
                const fmt=v=>key==="mon"?fK(v):Math.round(v);
                return <tr key={label} style={{background:ri%2?"#fafafa":"#fff",borderBottom:"1px solid #f1f5f9"}}>
                  <td style={{padding:"8px 12px",fontWeight:600,color:"#475569"}}>{label}</td>
                  <td style={{padding:"8px 12px",color:"#94a3b8"}}>{vals.length}</td>
                  <td style={{padding:"8px 12px"}}>{fmt(vals[0])}</td>
                  <td style={{padding:"8px 12px"}}>{fmt(vals[vals.length-1])}</td>
                  <td style={{padding:"8px 12px",fontWeight:600,color:"#6366f1"}}>{fmt(mean)}</td>
                  <td style={{padding:"8px 12px"}}>{fmt(median)}</td>
                  <td style={{padding:"8px 12px",color:"#94a3b8"}}>{fmt(std)}</td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ── TF Predictor ──
function TFPredictor({modelLoaded}){
  const [rec,setRec]=useState(30),[freq,setFreq]=useState(5),[mon,setMon]=useState(1500);
  const [result,setResult]=useState(null),[predicting,setPredicting]=useState(false);
  const predict=async()=>{
    setPredicting(true);setResult(null);
    const res=await predictSegmentTF(Number(rec),Number(freq),Number(mon));
    if(res){setResult(res);}else{
      const sc=v=>v>=.75?5:v>=.5?4:v>=.35?3:v>=.2?2:1;
      const rn=Math.max(0,Math.min(1,1-rec/400)),fn=Math.max(0,Math.min(1,freq/20)),mn2=Math.max(0,Math.min(1,mon/15000));
      const rs=sc(rn),fs=sc(fn),ms=sc(mn2);
      const seg=rs>=4&&fs>=4&&ms>=4?"VIP":(rs+fs+ms)/3>=3.5?"Leal":rs<=2&&(fs>=3||ms>=3)?"Em Risco":fs<=2&&ms<=2?"Inativo":"Novo";
      setResult({seg,rs,fs,ms,probs:null});
    }
    setPredicting(false);
  };
  const card={background:"#fff",borderRadius:14,border:"1px solid #e2e8f0",padding:"20px 22px"};
  const actions={VIP:"Programa de fidelidade exclusivo, ofertas personalizadas.",Leal:"Manter engajamento com comunicação regular.","Em Risco":"Campanha de reativação urgente, desconto especial.",Inativo:"Email de reengajamento com oferta agressiva.",Novo:"Onboarding personalizado, segunda compra incentivada."};
  return(
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      <div style={{...card,borderTop:"4px solid #6366f1"}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{fontSize:20}}>🤖</div>
          <div><div style={{fontSize:14,fontWeight:700}}>Preditor de Segmento — Deep Learning</div><div style={{fontSize:11,color:"#94a3b8"}}>{modelLoaded?"✅ TensorFlow.js ativo — modelo rodando no navegador":"⚠️ Usando regras RFM locais (modelo não encontrado)"}</div></div>
        </div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
        <div style={card}>
          <div style={{fontSize:13,fontWeight:700,marginBottom:16}}>Dados do Cliente</div>
          {[{label:"Recência (dias desde última compra)",val:rec,set:setRec,min:1,max:400,color:"#6366f1"},{label:"Frequência (nº de compras)",val:freq,set:setFreq,min:1,max:50,color:"#10b981"},{label:"Monetário (valor total R$)",val:mon,set:setMon,min:10,max:20000,color:"#f59e0b"}].map(({label,val,set,min,max,color})=>(
            <div key={label} style={{marginBottom:18}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}><span style={{fontSize:12,color:"#64748b"}}>{label}</span><span style={{fontSize:13,fontWeight:700,color}}>{val}</span></div>
              <input type="range" min={min} max={max} value={val} onChange={e=>set(e.target.value)} style={{width:"100%",accentColor:color}}/>
            </div>
          ))}
          <button onClick={predict} disabled={predicting} style={{width:"100%",padding:"11px",borderRadius:10,border:"none",background:"linear-gradient(135deg,#6366f1,#8b5cf6)",color:"#fff",fontSize:14,fontWeight:700,cursor:"pointer",opacity:predicting?.6:1}}>
            {predicting?"🔄 Processando...":"🚀 Prever Segmento com IA"}
          </button>
        </div>
        <div style={card}>
          <div style={{fontSize:13,fontWeight:700,marginBottom:16}}>Resultado da Predição</div>
          {result?(
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              <div style={{textAlign:"center",padding:"20px",borderRadius:12,background:SEGS[result.seg].bg,border:`2px solid ${SEGS[result.seg].hex}`}}>
                <div style={{fontSize:40,marginBottom:8}}>{SEGS[result.seg].emoji}</div>
                <div style={{fontSize:22,fontWeight:800,color:SEGS[result.seg].tx}}>{result.seg}</div>
                <div style={{fontSize:12,color:"#94a3b8",marginTop:4}}>{SEGS[result.seg].desc}</div>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8}}>
                {[["R Score",result.rs,"#6366f1"],["F Score",result.fs,"#10b981"],["M Score",result.ms,"#f59e0b"]].map(([l,v,c])=>(
                  <div key={l} style={{textAlign:"center",padding:"10px",borderRadius:8,background:"#f8fafc"}}>
                    <div style={{fontSize:10,color:"#94a3b8",marginBottom:4}}>{l}</div>
                    <div style={{fontSize:22,fontWeight:700,color:c}}>{v}</div>
                    <ScoreBar v={v}/>
                  </div>
                ))}
              </div>
              {result.probs&&<div>
                <div style={{fontSize:11,fontWeight:600,color:"#64748b",marginBottom:8}}>Confiança por Cluster (TF.js)</div>
                {result.probs.map((p,i)=><div key={i} style={{display:"flex",alignItems:"center",gap:8,marginBottom:5}}>
                  <span style={{fontSize:10,color:"#94a3b8",minWidth:54}}>Cluster {i}</span>
                  <MiniBar value={p} max={1} color={i===result.clusterIdx?"#6366f1":"#e2e8f0"}/>
                  <span style={{fontSize:11,fontWeight:600,color:i===result.clusterIdx?"#6366f1":"#94a3b8"}}>{(p*100).toFixed(1)}%</span>
                </div>)}
              </div>}
              <div style={{padding:"12px",borderRadius:8,background:"#eff6ff",border:"1px solid #bfdbfe",fontSize:12,color:"#1e3a8a"}}><strong>💡 Ação recomendada:</strong> {actions[result.seg]}</div>
            </div>
          ):(
            <div style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",height:260,color:"#94a3b8",gap:12}}>
              <div style={{fontSize:40}}>🎯</div><div style={{fontSize:13}}>Ajuste os sliders e clique em Prever</div>
            </div>
          )}
        </div>
      </div>
      <div style={{...card,background:"linear-gradient(135deg,#1e1b4b,#312e81)",color:"#fff"}}>
        <div style={{fontSize:13,fontWeight:700,marginBottom:8}}>🧠 Como funciona o Deep Learning aqui</div>
        <div style={{fontSize:12,color:"rgba(255,255,255,0.7)",lineHeight:1.7}}>
          O modelo foi treinado no Google Colab com os dados reais do Online Retail II (4.338 clientes). Após o treino, foi exportado para <strong style={{color:"#a5b4fc"}}>TensorFlow.js</strong> e carregado diretamente neste navegador — <strong style={{color:"#a5b4fc"}}>zero backend, zero servidor</strong>. A predição acontece 100% no seu dispositivo usando a GPU/CPU local via WebGL.
        </div>
      </div>
    </div>
  );
}

// ── APP ──
export default function App(){
  const [clients,setClients]=useState([]);
  const [filter,setFilter]=useState("Todos");
  const [page,setPage]=useState("overview");
  const [hov,setHov]=useState(null);
  const [sort,setSort]=useState({col:"mon",asc:false});
  const [tpage,setTpage]=useState(0);
  const [sideOpen,setSideOpen]=useState(true);
  const [loading,setLoading]=useState(true);
  const [modelLoaded,setModelLoaded]=useState(false);
  const PAGE=12;

  useEffect(()=>{
    setLoading(true);
    loadModel().then(ok=>setModelLoaded(ok));
    setTimeout(()=>{setClients(buildClients(genRows(180)));setLoading(false);},800);
  },[]);

  const handleFile=e=>{
    const f=e.target.files[0];if(!f)return;
    setLoading(true);
    const r=new FileReader();
    r.onload=ev=>{
      try{setClients(buildClients(parseCSV(ev.target.result)));setTpage(0);setFilter("Todos");}
      catch{alert("Erro ao processar CSV.\nFormato esperado: customer_id, purchase_date, purchase_value");}
      setLoading(false);
    };
    r.readAsText(f);
  };

  const counts=Object.fromEntries(SK.map(s=>[s,clients.filter(c=>c.segment===s).length]));
  const revs=Object.fromEntries(SK.map(s=>[s,clients.filter(c=>c.segment===s).reduce((a,c)=>a+c.mon,0)]));
  const totalRev=clients.reduce((a,c)=>a+c.mon,0);
  const fil=filter==="Todos"?clients:clients.filter(c=>c.segment===filter);
  const srtd=[...fil].sort((a,b)=>sort.asc?a[sort.col]-b[sort.col]:b[sort.col]-a[sort.col]);
  const tPages=Math.ceil(srtd.length/PAGE),pageD=srtd.slice(tpage*PAGE,(tpage+1)*PAGE);
  const top5=[...clients].sort((a,b)=>b.mon-a.mon).slice(0,5);

  const navItems=[
    {id:"overview",icon:"◉",label:"Visão Geral"},
    {id:"analytics",icon:"📊",label:"Análise Gráfica"},
    {id:"scatter",icon:"⬡",label:"Dispersão RFM"},
    {id:"clients",icon:"≡",label:"Clientes"},
    {id:"top",icon:"★",label:"Top Clientes"},
    {id:"predictor",icon:"🤖",label:"Preditor IA"},
  ];

  const badge=s=>({display:"inline-flex",alignItems:"center",gap:5,padding:"3px 10px",borderRadius:20,fontSize:11,fontWeight:600,background:SEGS[s].bg,color:SEGS[s].tx});
  const card={background:"#fff",borderRadius:14,border:"1px solid #e2e8f0",padding:"20px 22px"};

  if(loading)return(
    <div style={{display:"flex",height:"100vh",alignItems:"center",justifyContent:"center",flexDirection:"column",gap:16,fontFamily:"system-ui,sans-serif",background:"#f8fafc"}}>
      <div style={{width:48,height:48,borderRadius:14,background:"linear-gradient(135deg,#6366f1,#8b5cf6)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:24}}>📊</div>
      <div style={{fontSize:16,fontWeight:600,color:"#1e1b4b"}}>Carregando RFM Pro...</div>
      <div style={{width:200,height:4,borderRadius:2,background:"#e2e8f0",overflow:"hidden"}}><div style={{width:"70%",height:"100%",background:"linear-gradient(90deg,#6366f1,#8b5cf6)",borderRadius:2}}/></div>
    </div>
  );

  return(
    <div style={{display:"flex",height:"100vh",fontFamily:"system-ui,sans-serif",background:"#f8fafc",color:"#0f172a",fontSize:14}}>
      <div style={{width:sideOpen?220:64,flexShrink:0,background:"#1e1b4b",display:"flex",flexDirection:"column",transition:"width .25s",overflow:"hidden"}}>
        <div style={{padding:"20px 16px",display:"flex",alignItems:"center",gap:10,borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
          <div style={{width:34,height:34,borderRadius:10,background:"linear-gradient(135deg,#6366f1,#8b5cf6)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:16,flexShrink:0}}>📊</div>
          {sideOpen&&<div><div style={{fontSize:15,fontWeight:700,color:"#fff"}}>RFM<span style={{color:"#a5b4fc"}}>Pro</span></div><div style={{fontSize:10,color:"rgba(255,255,255,0.4)"}}>Analytics Platform</div></div>}
        </div>
        <div style={{flex:1,padding:"12px 0"}}>
          {sideOpen&&<div style={{padding:"6px 24px 10px",fontSize:10,fontWeight:700,color:"rgba(255,255,255,0.3)",letterSpacing:1,textTransform:"uppercase"}}>Menu</div>}
          {navItems.map(item=>{const act=page===item.id;return <div key={item.id} onClick={()=>setPage(item.id)} style={{display:"flex",alignItems:"center",gap:12,padding:"10px 16px",cursor:"pointer",borderRadius:8,margin:"2px 8px",background:act?"rgba(99,102,241,0.2)":"transparent",color:act?"#a5b4fc":"rgba(255,255,255,0.6)",transition:"all .15s",whiteSpace:"nowrap"}}><span style={{fontSize:16,flexShrink:0}}>{item.icon}</span>{sideOpen&&<span style={{fontSize:13,fontWeight:500}}>{item.label}</span>}</div>;})}
        </div>
        {sideOpen&&<div style={{padding:"12px 16px",borderTop:"1px solid rgba(255,255,255,0.08)"}}>
          <div style={{fontSize:10,color:"rgba(255,255,255,0.4)",marginBottom:6}}>Motor de IA</div>
          <div style={{display:"flex",alignItems:"center",gap:6}}><div style={{width:8,height:8,borderRadius:"50%",background:modelLoaded?"#10b981":"#f59e0b"}}/><span style={{fontSize:11,color:"rgba(255,255,255,0.6)"}}>{modelLoaded?"TF.js Ativo":"Regras RFM"}</span></div>
        </div>}
        <div style={{padding:12,borderTop:"1px solid rgba(255,255,255,0.08)"}}><div onClick={()=>setSideOpen(!sideOpen)} style={{display:"flex",alignItems:"center",justifyContent:"center",padding:"10px 16px",cursor:"pointer",borderRadius:8,color:"rgba(255,255,255,0.6)"}}><span style={{fontSize:14}}>{sideOpen?"◀":"▶"}</span></div></div>
      </div>

      <div style={{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"}}>
        <div style={{background:"#fff",borderBottom:"1px solid #e2e8f0",padding:"0 24px",height:60,display:"flex",alignItems:"center",gap:12,flexShrink:0,flexWrap:"wrap"}}>
          <div style={{flex:1}}><div style={{fontSize:16,fontWeight:700}}>{navItems.find(n=>n.id===page)?.label}</div><div style={{fontSize:12,color:"#94a3b8"}}>{clients.length} clientes · {new Date().toLocaleDateString("pt-BR",{day:"2-digit",month:"long",year:"numeric"})}</div></div>
          <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
            {["Todos",...SK].map(sg=>{const act=filter===sg,col=SEGS[sg]?.hex;return <button key={sg} onClick={()=>{setFilter(sg);setTpage(0);}} style={{padding:"5px 12px",borderRadius:20,border:act?`1.5px solid ${col||"#6366f1"}`:"1px solid #e2e8f0",background:act?(SEGS[sg]?.bg||"#eff6ff"):"#fff",color:act?(SEGS[sg]?.tx||"#1e3a8a"):"#64748b",fontSize:11,fontWeight:act?600:400,cursor:"pointer"}}>{SEGS[sg]?.emoji} {sg}{SEGS[sg]?` (${counts[sg]||0})`:""}</button>;})}
          </div>
          <label style={{padding:"7px 14px",borderRadius:8,fontSize:12,fontWeight:500,cursor:"pointer",border:"1px solid #e2e8f0",background:"#f8fafc",color:"#475569"}}>↑ CSV<input type="file" accept=".csv" onChange={handleFile} style={{display:"none"}}/></label>
          <button onClick={()=>exportCSV(clients,filter)} style={{padding:"7px 14px",borderRadius:8,fontSize:12,fontWeight:500,cursor:"pointer",border:"none",background:"#6366f1",color:"#fff"}}>↓ Exportar</button>
        </div>

        <div style={{flex:1,overflowY:"auto",padding:24}}>
          {page==="overview"&&<div style={{display:"flex",flexDirection:"column",gap:20}}>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(170px,1fr))",gap:14}}>
              {[{label:"Total Clientes",value:fN(clients.length),sub:"base ativa",color:"#6366f1",icon:"👥"},{label:"Receita Total",value:fK(totalRev),sub:"período analisado",color:"#10b981",icon:"💰"},{label:"Ticket Médio",value:fK(clients.length?totalRev/clients.length:0),sub:"por cliente",color:"#3b82f6",icon:"🎫"},{label:"Clientes VIP",value:fN(counts.VIP||0),sub:`${clients.length?(((counts.VIP||0)/clients.length)*100).toFixed(1):0}% da base`,color:"#8b5cf6",icon:"👑"},{label:"Em Risco",value:fN(counts["Em Risco"]||0),sub:"precisam atenção",color:"#f59e0b",icon:"⚠️"}].map(k=>(
                <div key={k.label} style={{background:"#fff",borderRadius:14,border:"1px solid #e2e8f0",padding:"18px 20px",borderLeft:`4px solid ${k.color}`}}>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:10}}><div style={{fontSize:11,fontWeight:600,color:"#64748b",textTransform:"uppercase",letterSpacing:.5}}>{k.label}</div><div style={{fontSize:18}}>{k.icon}</div></div>
                  <div style={{fontSize:26,fontWeight:700,color:k.color,marginBottom:2}}>{k.value}</div>
                  <div style={{fontSize:11,color:"#94a3b8"}}>{k.sub}</div>
                </div>
              ))}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:12}}>
              {SK.map(sg=>{const cnt=counts[sg]||0,rev=revs[sg]||0,pct=clients.length?cnt/clients.length*100:0,act=filter===sg;return <div key={sg} onClick={()=>setFilter(act?"Todos":sg)} style={{background:act?SEGS[sg].bg:"#fff",borderRadius:12,border:`${act?"1.5px":"1px"} solid ${act?SEGS[sg].hex+"66":"#e2e8f0"}`,padding:"14px 16px",cursor:"pointer",transition:"all .15s"}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}><span style={{fontSize:20}}>{SEGS[sg].emoji}</span><span style={{fontSize:11,fontWeight:700,color:SEGS[sg].hex}}>{pct.toFixed(1)}%</span></div>
                <div style={{fontSize:13,fontWeight:700,color:SEGS[sg].tx,marginBottom:2}}>{sg}</div>
                <div style={{fontSize:11,color:"#94a3b8",marginBottom:10}}>{SEGS[sg].desc}</div>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}><MiniBar value={cnt} max={clients.length||1} color={SEGS[sg].hex}/><span style={{fontSize:11,fontWeight:600,color:"#475569",minWidth:24}}>{cnt}</span></div>
                <div style={{fontSize:11,color:"#64748b"}}>Receita: <strong style={{color:SEGS[sg].hex}}>{fK(rev)}</strong></div>
              </div>;})}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
              <div style={card}>
                <div style={{fontSize:13,fontWeight:700,marginBottom:16,display:"flex",justifyContent:"space-between"}}><span>Top 5 Clientes</span><span style={{fontSize:11,color:"#94a3b8"}}>por receita</span></div>
                {top5.map((c,i)=><div key={c.id} style={{display:"flex",alignItems:"center",gap:12,padding:"8px 0",borderBottom:i<4?"1px solid #f8fafc":"none"}}>
                  <div style={{width:26,height:26,borderRadius:8,background:i===0?"#fef3c7":i===1?"#f1f5f9":"#f8fafc",display:"flex",alignItems:"center",justifyContent:"center",fontSize:11,fontWeight:700,color:i===0?"#d97706":"#64748b",flexShrink:0}}>{i+1}</div>
                  <div style={{flex:1}}><div style={{fontSize:12,fontWeight:600,fontFamily:"monospace"}}>{c.id}</div><span style={badge(c.segment)}>{SEGS[c.segment].emoji} {c.segment}</span></div>
                  <div style={{fontSize:13,fontWeight:700,color:"#10b981"}}>{fK(c.mon)}</div>
                </div>)}
              </div>
              <div style={card}>
                <div style={{fontSize:13,fontWeight:700,marginBottom:16}}>Risco de Churn por Segmento</div>
                {SK.map(sg=>{const churnVal=CHURN_MAP[SEGS[sg].hex]||0.3;return <div key={sg} style={{display:"flex",alignItems:"center",gap:12,marginBottom:12}}><div style={{fontSize:13,minWidth:80,color:SEGS[sg].tx,fontWeight:500}}>{SEGS[sg].emoji} {sg}</div><MiniBar value={churnVal} max={1} color={SEGS[sg].hex}/><div style={{fontSize:12,fontWeight:600,color:SEGS[sg].hex,minWidth:36}}>{(churnVal*100).toFixed(0)}%</div></div>;})}
              </div>
            </div>
          </div>}

          {page==="analytics"&&<AnalyticsPage clients={clients} filter={filter}/>}

          {page==="scatter"&&<div style={{display:"flex",flexDirection:"column",gap:16}}>
            <div style={card}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16,flexWrap:"wrap",gap:8}}>
                <div><div style={{fontSize:13,fontWeight:700}}>Dispersão RFM</div><div style={{fontSize:11,color:"#94a3b8"}}>X = Recência · Y = Monetário · Tamanho = Frequência</div></div>
                <div style={{display:"flex",gap:12,flexWrap:"wrap"}}>{SK.map(sg=><span key={sg} style={{fontSize:11,display:"flex",alignItems:"center",gap:4,color:"#64748b"}}><span style={{width:8,height:8,borderRadius:2,background:SEGS[sg].hex,display:"inline-block"}}/>{sg}</span>)}</div>
              </div>
              <ScatterCanvas data={clients} filter={filter} hovId={hov?.id} setHov={setHov}/>
            </div>
            {hov&&<div style={{...card,borderLeft:`4px solid ${SEGS[hov.segment].hex}`,display:"flex",gap:20,alignItems:"center",flexWrap:"wrap"}}>
              <ChurnRing value={hov.churn}/>
              <div style={{flex:1}}>
                <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:8}}><span style={{fontSize:15,fontWeight:700,fontFamily:"monospace"}}>{hov.id}</span><span style={badge(hov.segment)}>{SEGS[hov.segment].emoji} {hov.segment}</span></div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12}}>{[["Recência",`${hov.rec} dias`],["Frequência",`${hov.freq} compras`],["Monetário",fR(hov.mon)]].map(([l,v])=><div key={l} style={{background:"#f8fafc",borderRadius:8,padding:"8px 10px"}}><div style={{fontSize:10,color:"#94a3b8",marginBottom:2}}>{l}</div><div style={{fontSize:13,fontWeight:700}}>{v}</div></div>)}</div>
              </div>
              <div style={{display:"flex",flexDirection:"column",gap:6}}>{[["R",hov.rs],["F",hov.fs],["M",hov.ms]].map(([l,v])=><div key={l} style={{display:"flex",alignItems:"center",gap:8}}><span style={{fontSize:11,color:"#94a3b8",width:14}}>{l}</span><ScoreBar v={v}/><span style={{fontSize:11,fontWeight:700,color:"#475569"}}>{v}</span></div>)}</div>
            </div>}
          </div>}

          {page==="clients"&&<div style={card}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16}}>
              <div><div style={{fontSize:13,fontWeight:700}}>Base de Clientes</div><div style={{fontSize:11,color:"#94a3b8"}}>{fil.length} clientes</div></div>
              <button onClick={()=>exportCSV(clients,filter)} style={{padding:"7px 14px",borderRadius:8,fontSize:12,fontWeight:500,cursor:"pointer",border:"1px solid #e2e8f0",background:"#f8fafc",color:"#475569"}}>↓ Exportar seleção</button>
            </div>
            <div style={{overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse"}}>
                <thead><tr style={{borderBottom:"2px solid #f1f5f9"}}>{[["id","Cliente"],["segment","Segmento"],["rec","Recência"],["freq","Compras"],["mon","Valor"],["churn","Churn"],["rs","R"],["fs","F"],["ms","M"]].map(([col,lb])=><th key={col} onClick={()=>{if(col==="segment"||col==="churn")return;setSort(st=>({col,asc:st.col===col?!st.asc:false}));setTpage(0);}} style={{padding:"10px 12px",textAlign:"left",fontSize:11,fontWeight:600,color:"#64748b",cursor:col!=="segment"&&col!=="churn"?"pointer":"default",whiteSpace:"nowrap",userSelect:"none",background:sort.col===col?"#f8fafc":"transparent"}}>{lb}{sort.col===col?(sort.asc?" ↑":" ↓"):""}</th>)}</tr></thead>
                <tbody>{pageD.map((c,i)=><tr key={c.id} style={{background:i%2?"#fafafa":"#fff",borderBottom:"1px solid #f1f5f9"}}><td style={{padding:"9px 12px",fontFamily:"monospace",fontSize:11,color:"#64748b"}}>{c.id}</td><td style={{padding:"9px 12px"}}><span style={badge(c.segment)}>{SEGS[c.segment].emoji} {c.segment}</span></td><td style={{padding:"9px 12px"}}>{c.rec}d</td><td style={{padding:"9px 12px"}}>{c.freq}x</td><td style={{padding:"9px 12px",fontWeight:600}}>{fR(c.mon)}</td><td style={{padding:"9px 12px"}}><span style={{fontSize:12,fontWeight:700,color:parseFloat(c.churn)>.5?"#ef4444":parseFloat(c.churn)>.3?"#f59e0b":"#10b981"}}>{(parseFloat(c.churn)*100).toFixed(0)}%</span></td><td style={{padding:"9px 12px"}}><ScoreBar v={c.rs}/></td><td style={{padding:"9px 12px"}}><ScoreBar v={c.fs}/></td><td style={{padding:"9px 12px"}}><ScoreBar v={c.ms}/></td></tr>)}</tbody>
              </table>
            </div>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:16,fontSize:12,color:"#64748b"}}>
              <span>Página {tpage+1} de {tPages||1} · {fil.length} registros</span>
              <div style={{display:"flex",gap:8}}>{["← Anterior","Próxima →"].map((lb,di)=>{const dis=di===0?tpage===0:tpage>=tPages-1;return <button key={lb} onClick={()=>setTpage(p=>di===0?Math.max(0,p-1):Math.min(tPages-1,p+1))} disabled={dis} style={{padding:"6px 12px",border:"1px solid #e2e8f0",borderRadius:8,background:"transparent",cursor:dis?"not-allowed":"pointer",opacity:dis?.4:1,fontSize:12}}>{lb}</button>;})}</div>
            </div>
          </div>}

          {page==="top"&&<div style={{display:"flex",flexDirection:"column",gap:14}}>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:12}}>
              {[...clients].sort((a,b)=>b.mon-a.mon).slice(0,3).map((c,i)=>(
                <div key={c.id} style={{...card,borderTop:`4px solid ${i===0?"#f59e0b":i===1?"#94a3b8":"#cd7c3f"}`,textAlign:"center"}}>
                  <div style={{fontSize:28,marginBottom:8}}>{i===0?"🥇":i===1?"🥈":"🥉"}</div>
                  <div style={{fontSize:16,fontWeight:700,fontFamily:"monospace",marginBottom:6}}>{c.id}</div>
                  <span style={badge(c.segment)}>{SEGS[c.segment].emoji} {c.segment}</span>
                  <div style={{fontSize:22,fontWeight:700,color:"#10b981",margin:"12px 0 4px"}}>{fR(c.mon)}</div>
                  <div style={{fontSize:11,color:"#94a3b8"}}>{c.freq} compras · {c.rec} dias</div>
                </div>
              ))}
            </div>
            <div style={card}>
              <div style={{fontSize:13,fontWeight:700,marginBottom:16}}>Top 10 Clientes por Receita</div>
              <table style={{width:"100%",borderCollapse:"collapse"}}>
                <thead><tr style={{borderBottom:"2px solid #f1f5f9"}}>{["#","Cliente","Segmento","Recência","Compras","Receita","Churn"].map(h=><th key={h} style={{padding:"10px 14px",textAlign:"left",fontSize:11,fontWeight:600,color:"#64748b",whiteSpace:"nowrap"}}>{h}</th>)}</tr></thead>
                <tbody>{[...clients].sort((a,b)=>b.mon-a.mon).slice(0,10).map((c,i)=><tr key={c.id} style={{background:i%2?"#fafafa":"#fff",borderBottom:"1px solid #f1f5f9"}}><td style={{padding:"10px 14px",fontWeight:700,color:i<3?"#d97706":"#94a3b8"}}>{i+1}</td><td style={{padding:"10px 14px",fontFamily:"monospace",fontSize:12}}>{c.id}</td><td style={{padding:"10px 14px"}}><span style={badge(c.segment)}>{SEGS[c.segment].emoji} {c.segment}</span></td><td style={{padding:"10px 14px"}}>{c.rec}d</td><td style={{padding:"10px 14px"}}>{c.freq}x</td><td style={{padding:"10px 14px",fontWeight:700,color:"#10b981"}}>{fR(c.mon)}</td><td style={{padding:"10px 14px"}}><ChurnRing value={c.churn}/></td></tr>)}</tbody>
              </table>
            </div>
          </div>}

          {page==="predictor"&&<TFPredictor modelLoaded={modelLoaded}/>}
        </div>
      </div>
    </div>
  );
}

 
