import { lazy, Suspense } from "react";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { AppProviders } from "./providers";
import { RuntimeModeBanner } from "../features/ai-tuning/components/RuntimeModeBanner";
import { useRuntimeQueries } from "../features/ai-tuning/hooks/useRuntimeQueries";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/600.css";
import "@fontsource/newsreader/600.css";
import "../styles/tokens.css";
import "../styles/global.css";

const OverviewPage = lazy(() => import("../features/ai-tuning/pages/OverviewPage").then((module) => ({ default: module.OverviewPage })));
const CreateTaskPage = lazy(() => import("../features/ai-tuning/pages/CreateTaskPage").then((module) => ({ default: module.CreateTaskPage })));
const CampaignDetailPage = lazy(() => import("../features/ai-tuning/pages/CampaignDetailPage").then((module) => ({ default: module.CampaignDetailPage })));
const CampaignResultPage = lazy(() => import("../features/ai-tuning/pages/CampaignResultPage").then((module) => ({ default: module.CampaignResultPage })));
const OptimizationRunPage = lazy(() => import("../features/ai-tuning/pages/OptimizationRunPage").then((module) => ({ default: module.OptimizationRunPage })));
const SweepRunPage = lazy(() => import("../features/ai-tuning/pages/SweepRunPage").then((module) => ({ default: module.SweepRunPage })));
const RecoveryPage = lazy(() => import("../features/ai-tuning/pages/RecoveryPage").then((module) => ({ default: module.RecoveryPage })));

function ApplicationFrame() {
  const { capabilities, campaigns } = useRuntimeQueries();
  const active = campaigns.data?.find((campaign) => !["COMPLETED", "FAILED"].includes(campaign.state)) ?? null;
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">跳到主要内容 / Skip to content</a>
      <RuntimeModeBanner capability={capabilities.data} activeCampaign={active} />
      <header className="topbar"><NavLink className="brand-lockup" to="/"><span className="brand-mark">V</span><span><b>VCSEL / AI TUNING</b><small>PHOTON LAB · CONTROL SURFACE</small></span></NavLink><span className="api-mark">PRODUCT 1.1.0 · SCHEMA V1.1</span></header>
      <div className="workspace">
        <aside className="sidebar" aria-label="Primary navigation"><nav><NavLink to="/" end><span>Overview</span><small>总览</small></NavLink><NavLink to="/tasks/new"><span>New task</span><small>新建任务</small></NavLink><a href="#recent"><span>History</span><small>历史记录</small></a></nav><div className="safety-note"><b>FASTAPI CONTROL ONLY</b><span>No raw commands<br />No direct hardware access<br />无浏览器直连硬件</span></div></aside>
        <main id="main-content" tabIndex={-1}><Suspense fallback={<div className="panel skeleton">Loading control surface… / 正在载入…</div>}><Routes><Route path="/" element={<OverviewPage />} /><Route path="/tasks/new" element={<CreateTaskPage />} /><Route path="/runs/optimization/:campaignId" element={<OptimizationRunPage />} /><Route path="/runs/sweep/:campaignId" element={<SweepRunPage />} /><Route path="/recovery/:campaignId" element={<RecoveryPage />} /><Route path="/campaigns/:campaignId" element={<CampaignDetailPage />} /><Route path="/campaigns/:campaignId/result" element={<CampaignResultPage />} /></Routes></Suspense></main>
      </div>
    </div>
  );
}

export function App() { return <AppProviders><BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><ApplicationFrame /></BrowserRouter></AppProviders>; }
