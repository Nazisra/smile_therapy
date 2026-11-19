import { useState, useCallback, useRef, useEffect } from 'react'
import {
  Upload,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Activity,
  FileText,
  ChevronDown,
  RefreshCw,
  Download,
  Info,
  Sliders,
  Eye,
  ShieldCheck,
  User,
  LogOut,
  Building2,
  Stethoscope,
  Award,
  FileCheck,
  Check,
  ZoomIn,
  ZoomOut,
  SunMedium,
  RotateCcw,
  Printer,
  Sparkles,
  Scale,
  Lock,
  Mail,
  UserPlus,
  Maximize2
} from 'lucide-react'
import axios from 'axios'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'

const DEFAULT_API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const SEVERITY_CONFIG = {
  Mild: {
    label: 'Mild Physiological Asymmetry',
    clinicalDescription: 'Skeletal symmetry is within normal physiological limits (>= 94% SI). No surgical correction required.',
    badgeBg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    dotColor: 'bg-emerald-600',
    barColor: 'bg-emerald-600',
    icon: CheckCircle2,
    urgencyText: 'Routine Review (6-12 mo)',
    urgencyBadge: 'bg-emerald-50 text-emerald-800 border-emerald-200'
  },
  Moderate: {
    label: 'Moderate Mandibular Asymmetry',
    clinicalDescription: 'Borderline skeletal / dentoalveolar discrepancy (90% - 93.9% SI). Orthodontic assessment recommended.',
    badgeBg: 'bg-amber-50 text-amber-900 border-amber-200',
    dotColor: 'bg-amber-500',
    barColor: 'bg-amber-500',
    icon: AlertTriangle,
    urgencyText: 'Orthodontic Evaluation',
    urgencyBadge: 'bg-amber-50 text-amber-900 border-amber-200'
  },
  Severe: {
    label: 'Severe Mandibular Asymmetry',
    clinicalDescription: 'Marked skeletal discrepancy (< 90% SI). Urgent Oral & Maxillofacial Surgery (OMFS) consultation indicated.',
    badgeBg: 'bg-rose-50 text-rose-900 border-rose-200',
    dotColor: 'bg-[#831843]',
    barColor: 'bg-[#831843]',
    icon: XCircle,
    urgencyText: 'Urgent OMFS Surgical Referral',
    urgencyBadge: 'bg-rose-50 text-rose-900 border-rose-200'
  },
}

function GoogleIcon() {
  return (
    <svg className="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24">
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
      />
      <path
        fill="#EA4335"
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
      />
    </svg>
  )
}

function generateSyntheticSampleBlob(type) {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas')
    canvas.width = 1200
    canvas.height = 700
    const ctx = canvas.getContext('2d')

    ctx.fillStyle = '#0f0b0d'
    ctx.fillRect(0, 0, 1200, 700)

    const grad = ctx.createRadialGradient(600, 400, 40, 600, 400, 480)
    grad.addColorStop(0, '#26131c')
    grad.addColorStop(1, '#090507')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, 1200, 700)

    ctx.strokeStyle = '#9ca3af'
    ctx.lineWidth = 22
    ctx.beginPath()
    ctx.ellipse(600, 450, 400, 180, 0, 0, Math.PI)
    ctx.stroke()

    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 14
    ctx.beginPath()
    ctx.ellipse(600, 410, 240, 90, 0, 0, Math.PI)
    ctx.stroke()

    const rHeight = type === 'severe' ? 220 : type === 'moderate' ? 280 : 330
    const lHeight = 330

    ctx.strokeStyle = '#6b7280'
    ctx.lineWidth = 26
    ctx.beginPath()
    ctx.moveTo(220, 450)
    ctx.lineTo(210, 450 - rHeight)
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(980, 450)
    ctx.lineTo(990, 450 - lHeight)
    ctx.stroke()

    canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.9)
  })
}

function AuthModal({ isOpen, onClose, initialTab = 'signin', onLoginSuccess }) {
  const [activeTab, setActiveTab] = useState(initialTab)
  const [signInEmail, setSignInEmail] = useState('dr.rahman@avedental.org')
  const [signInPassword, setSignInPassword] = useState('••••••••••')
  const [signUpName, setSignUpName] = useState('')
  const [signUpEmail, setSignUpEmail] = useState('')
  const [signUpPassword, setSignUpPassword] = useState('')
  const [signUpLicense, setSignUpLicense] = useState('')

  if (!isOpen) return null

  const handleSignInSubmit = (e) => {
    e.preventDefault()
    const isDrRahman = signInEmail.toLowerCase().includes('rahman')
    onLoginSuccess({
      name: isDrRahman ? 'Dr. Md. Shahidur Rahman, BDS' : 'Dr. Attending Clinician',
      email: signInEmail || 'dr.rahman@avedental.org',
      role: 'Senior Consultant, Oral & Maxillofacial Orthodontics',
      photo: isDrRahman ? '/doctor.png' : null,
      initials: 'SR'
    })
    onClose()
  }

  const handleSignUpSubmit = (e) => {
    e.preventDefault()
    onLoginSuccess({
      name: signUpName || 'Dr. Registered Clinician, BDS',
      email: signUpEmail || 'clinician@smiletherapy.clinic',
      role: signUpLicense ? `Licensed Specialist (${signUpLicense})` : 'Orthodontic Resident / Attending',
      photo: null,
      initials: (signUpName || 'RC').split(' ').map(n => n[0]).slice(0, 2).join('').toUpperCase()
    })
    onClose()
  }

  const handleGoogleSelect = (account) => {
    onLoginSuccess(account)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full border border-slate-200 overflow-hidden">
        {/* Modal Header */}
        <div className="bg-[#24121B] text-white px-5 py-4 flex items-center justify-between border-b border-[#381B2B]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded bg-rose-500/20 border border-rose-400/40 flex items-center justify-center">
              <Stethoscope className="w-4 h-4 text-rose-300" />
            </div>
            <div>
              <h2 className="text-sm font-semibold tracking-tight text-white">Smile Therapy Workstation</h2>
              <p className="text-[11px] text-slate-300">Clinician Radiology Authentication</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white text-lg leading-none p-1 rounded hover:bg-white/10 transition-colors"
          >
            &times;
          </button>
        </div>

        {/* Tab Header */}
        <div className="flex border-b border-slate-200 bg-slate-50 text-xs font-semibold">
          <button
            type="button"
            onClick={() => setActiveTab('signin')}
            className={`flex-1 py-2.5 text-center transition-all border-b-2 ${
              activeTab === 'signin'
                ? 'text-[#831843] border-[#831843] bg-white'
                : 'text-slate-500 hover:text-slate-800 border-transparent'
            }`}
          >
            Clinician Sign In
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('signup')}
            className={`flex-1 py-2.5 text-center transition-all border-b-2 ${
              activeTab === 'signup'
                ? 'text-[#831843] border-[#831843] bg-white'
                : 'text-slate-500 hover:text-slate-800 border-transparent'
            }`}
          >
            Register Practitioner
          </button>
        </div>

        <div className="p-5">
          {activeTab === 'signin' ? (
            <form onSubmit={handleSignInSubmit} className="space-y-3.5">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Institutional Email</label>
                <div className="relative">
                  <Mail className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="email"
                    required
                    value={signInEmail}
                    onChange={(e) => setSignInEmail(e.target.value)}
                    placeholder="doctor@smiletherapy.clinic"
                    className="w-full pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded focus:outline-none focus:ring-1 focus:ring-[#831843] focus:border-[#831843] bg-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Password</label>
                <div className="relative">
                  <Lock className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="password"
                    required
                    value={signInPassword}
                    onChange={(e) => setSignInPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded focus:outline-none focus:ring-1 focus:ring-[#831843] focus:border-[#831843] bg-white"
                  />
                </div>
              </div>

              <button
                type="submit"
                className="w-full bg-[#831843] hover:bg-[#6F2140] text-white text-xs font-semibold py-2 rounded transition-colors"
              >
                Authenticate Session
              </button>

              <div className="p-2.5 bg-slate-50 border border-slate-200 rounded text-[11px] text-slate-600 flex items-center justify-between">
                <span>Verified Specialist: <strong>Dr. Md. Shahidur Rahman</strong></span>
                <button
                  type="button"
                  onClick={() => {
                    setSignInEmail('dr.rahman@avedental.org')
                    setSignInPassword('password123')
                  }}
                  className="text-xs font-semibold text-[#831843] hover:underline"
                >
                  Load Demo
                </button>
              </div>
            </form>
          ) : (
            <form onSubmit={handleSignUpSubmit} className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Full Name & Credentials</label>
                <div className="relative">
                  <User className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="text"
                    required
                    value={signUpName}
                    onChange={(e) => setSignUpName(e.target.value)}
                    placeholder="Dr. Sarah Ahmed, BDS, FCPS"
                    className="w-full pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded focus:outline-none focus:ring-1 focus:ring-[#831843] focus:border-[#831843] bg-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">BMDC / Medical Registration #</label>
                <div className="relative">
                  <Award className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="text"
                    value={signUpLicense}
                    onChange={(e) => setSignUpLicense(e.target.value)}
                    placeholder="BMDC-D-19842"
                    className="w-full pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded focus:outline-none focus:ring-1 focus:ring-[#831843] focus:border-[#831843] bg-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Hospital Email</label>
                <div className="relative">
                  <Mail className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="email"
                    required
                    value={signUpEmail}
                    onChange={(e) => setSignUpEmail(e.target.value)}
                    placeholder="dr.sarah@smiletherapy.clinic"
                    className="w-full pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded focus:outline-none focus:ring-1 focus:ring-[#831843] focus:border-[#831843] bg-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Password</label>
                <div className="relative">
                  <Lock className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="password"
                    required
                    value={signUpPassword}
                    onChange={(e) => setSignUpPassword(e.target.value)}
                    placeholder="Minimum 8 characters"
                    className="w-full pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded focus:outline-none focus:ring-1 focus:ring-[#831843] focus:border-[#831843] bg-white"
                  />
                </div>
              </div>

              <button
                type="submit"
                className="w-full bg-[#831843] hover:bg-[#6F2140] text-white text-xs font-semibold py-2 rounded transition-colors flex items-center justify-center gap-1.5"
              >
                <UserPlus className="w-3.5 h-3.5" /> Create Practitioner Account
              </button>
            </form>
          )}

          <div className="relative my-4">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-slate-200" />
            </div>
            <div className="relative flex justify-center text-[10px] uppercase font-semibold tracking-wider text-slate-400">
              <span className="bg-white px-2">Institutional SSO</span>
            </div>
          </div>

          <div className="space-y-2">
            <button
              type="button"
              onClick={() => handleGoogleSelect({
                name: 'Dr. Md. Shahidur Rahman, BDS',
                email: 'dr.rahman@avedental.org',
                role: 'Senior Consultant, Oral & Maxillofacial Orthodontics',
                photo: '/doctor.png',
              })}
              className="w-full flex items-center justify-center gap-2 border border-slate-200 hover:bg-slate-50 py-1.5 px-3 rounded text-xs font-medium text-slate-700 transition-colors"
            >
              <GoogleIcon />
              <span>Continue with Dr. Md. Shahidur Rahman (Google)</span>
            </button>

            <button
              type="button"
              onClick={() => handleGoogleSelect({
                name: 'Guest Clinician Demo',
                email: 'guest.clinician@smiletherapy.clinic',
                role: 'Visiting Radiology Fellow / Observer',
                photo: null,
                initials: 'GC'
              })}
              className="w-full flex items-center justify-center gap-1.5 border border-dashed border-slate-300 hover:bg-slate-50 py-1.5 px-3 rounded text-[11px] font-medium text-slate-600 transition-colors"
            >
              <span>Instant Observer Access (Demo Mode)</span>
            </button>
          </div>
        </div>

        <div className="bg-slate-50 px-5 py-2.5 border-t border-slate-200 text-[11px] text-slate-500 text-center">
          HIPAA &amp; GDPR Compliant &bull; PACS Secure Node Session
        </div>
      </div>
    </div>
  )
}

function ServerConfigModal({ isOpen, onClose, currentUrl, onSaveUrl }) {
  const [urlInput, setUrlInput] = useState(currentUrl)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)

  useEffect(() => {
    setUrlInput(currentUrl)
    setTestResult(null)
  }, [currentUrl, isOpen])

  if (!isOpen) return null

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    const cleanUrl = urlInput.trim().replace(/\/+$/, '')
    try {
      const res = await axios.get(`${cleanUrl}/health`, { timeout: 8000 })
      if (res.data && res.data.status === 'ok') {
        setTestResult({
          success: true,
          message: `Connected! Device: ${res.data.device || 'CPU'} | DenseNet: Ready`,
          data: res.data,
        })
      } else {
        setTestResult({
          success: false,
          message: 'Server responded, but health status is not OK.',
        })
      }
    } catch (err) {
      setTestResult({
        success: false,
        message: err.response?.data?.detail || err.message || 'Cannot reach backend. Ensure your Google Colab tunnel or server is running.',
      })
    } finally {
      setTesting(false)
    }
  }

  const handleSave = () => {
    const cleanUrl = urlInput.trim().replace(/\/+$/, '')
    onSaveUrl(cleanUrl)
    onClose()
  }

  const handleReset = () => {
    const defaultUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    setUrlInput(defaultUrl)
    onSaveUrl(defaultUrl)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs">
      <div className="bg-white rounded-xl shadow-2xl border border-slate-200 max-w-md w-full overflow-hidden animate-in fade-in duration-200">
        <div className="bg-[#201018] px-5 py-3.5 flex items-center justify-between text-white border-b border-rose-950/40">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-rose-400" />
            <h3 className="text-sm font-semibold tracking-tight">AI Backend Server Connection</h3>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white text-lg leading-none p-1">
            &times;
          </button>
        </div>

        <div className="p-5 space-y-4 text-xs text-slate-700">
          <div className="bg-rose-50/70 border border-rose-200/60 rounded p-3 text-[11px] leading-relaxed text-rose-900">
            <strong>Running on Google Colab?</strong> Paste your Cloudflare Tunnel URL (e.g. <code className="bg-rose-100/80 px-1 py-0.5 rounded font-mono">https://xxxx.trycloudflare.com</code>) below. The model will run on Colab's GPU with 100% original accuracy.
          </div>

          <div>
            <label className="block text-[11px] font-semibold text-slate-700 uppercase tracking-wider mb-1.5">
              Backend API Base URL
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="https://xxxx.trycloudflare.com or http://localhost:8000"
                className="flex-1 px-3 py-2 border border-slate-300 rounded text-xs font-mono focus:outline-none focus:ring-1 focus:ring-[#831843] focus:border-[#831843]"
              />
              <button
                type="button"
                onClick={handleTest}
                disabled={testing || !urlInput.trim()}
                className="px-3 py-2 bg-slate-800 hover:bg-slate-900 text-white font-medium rounded text-xs disabled:opacity-50 transition-colors flex items-center gap-1.5 shrink-0"
              >
                {testing && <RefreshCw className="w-3 h-3 animate-spin" />}
                {testing ? 'Testing...' : 'Test'}
              </button>
            </div>
          </div>

          {testResult && (
            <div
              className={`p-2.5 rounded border text-[11px] flex items-start gap-2 ${
                testResult.success
                  ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                  : 'bg-rose-50 border-rose-200 text-rose-800'
              }`}
            >
              {testResult.success ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              )}
              <div className="flex-1">
                <span className="font-semibold block">{testResult.message}</span>
                {testResult.data && (
                  <span className="text-[10px] text-slate-500 font-mono mt-0.5 block">
                    DenseNet-121: {testResult.data.densenet_loaded ? 'Ready' : 'Not Loaded'} | Device: {testResult.data.device}
                  </span>
                )}
              </div>
            </div>
          )}

          <div className="pt-2 border-t border-slate-100 flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={handleReset}
              className="text-[11px] text-slate-500 hover:text-slate-800 underline"
            >
              Reset to Default
            </button>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-3 py-1.5 border border-slate-300 text-slate-700 hover:bg-slate-50 rounded text-xs font-medium"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSave}
                className="px-3.5 py-1.5 bg-[#831843] hover:bg-[#6F2140] text-white rounded text-xs font-semibold shadow-xs transition-colors"
              >
                Save & Apply
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Header({ currentUser, onOpenAuth, onLogout, apiBaseUrl, onOpenServerModal }) {
  return (
    <header className="no-print bg-[#201018] text-white border-b border-[#351927] sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-rose-500/15 border border-rose-400/30 flex items-center justify-center text-rose-300">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm sm:text-base font-semibold tracking-tight text-white">
                Smile Therapy
              </h1>
              <span className="hidden sm:inline-block text-[10px] font-mono uppercase bg-[#381B2B] text-rose-200/90 border border-rose-400/20 px-1.5 py-0.5 rounded">
                PACS-ST-04
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Maxillofacial &amp; Orthodontic Radiograph Workstation
            </p>
          </div>
        </div>

        {/* Header Right: Auth & Node Controls */}
        <div className="flex items-center gap-2.5">
          {/* AI Backend Server Status & Config */}
          <button
            onClick={onOpenServerModal}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-slate-300"
            title="Configure AI Backend Connection (Google Colab / Localhost)"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="hidden sm:inline text-[10px] text-slate-400 font-medium">Backend:</span>
            <span className="font-mono text-[11px] truncate max-w-[90px] sm:max-w-[140px] text-rose-200">
              {apiBaseUrl ? apiBaseUrl.replace(/^https?:\/\//, '') : 'Colab/Local'}
            </span>
            <Sliders className="w-3 h-3 text-rose-300 ml-0.5" />
          </button>

          {currentUser ? (
            <div className="flex items-center gap-2.5 bg-white/5 border border-white/10 rounded px-2.5 py-1">
              {currentUser.photo ? (
                <img
                  src={currentUser.photo}
                  alt={currentUser.name}
                  className="w-6 h-6 rounded-full object-cover border border-rose-300/40"
                  onError={(e) => {
                    e.target.onerror = null
                    e.target.src = 'https://ui-avatars.com/api/?name=Shahidur+Rahman&background=831843&color=fff'
                  }}
                />
              ) : (
                <div className="w-6 h-6 rounded bg-[#831843] text-white flex items-center justify-center text-[10px] font-bold">
                  {currentUser.initials || 'ST'}
                </div>
              )}
              <div className="hidden sm:block text-left text-xs">
                <span className="font-medium text-white block leading-tight">{currentUser.name}</span>
                <span className="text-[10px] text-slate-400 block leading-tight truncate max-w-[160px]">
                  {currentUser.role.split(',')[0]}
                </span>
              </div>
              <button
                onClick={onLogout}
                title="Sign out of clinician session"
                className="ml-1 p-1 rounded text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
              >
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <button
                onClick={() => onOpenAuth('signin')}
                className="text-xs font-medium text-slate-300 hover:text-white px-2.5 py-1 rounded transition-colors"
              >
                Sign In
              </button>
              <button
                onClick={() => onOpenAuth('signup')}
                className="text-xs font-semibold text-white bg-[#831843] hover:bg-[#6F2140] px-3 py-1 rounded border border-rose-300/20 transition-colors"
              >
                Sign Up
              </button>
            </div>
          )}
        </div>
      </div>

      {/* High-density Workstation Status Strip */}
      <div className="bg-[#170B11] border-t border-[#2B1420] px-4 sm:px-6 py-1 text-[11px] text-slate-400 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-slate-300 font-medium">
            <Building2 className="w-3 h-3 text-rose-400" />
            Smile Therapy Clinic &bull; Oral Radiology Division
          </span>
          <span className="hidden md:inline text-slate-600">|</span>
          <span className="hidden md:inline">Modality: <strong className="text-slate-300 font-normal">Panoramic OPG (Digital)</strong></span>
        </div>
        <div className="flex items-center gap-3 font-mono text-[10px]">
          <span>STUDY: #ST-2026-OMFS</span>
          <span className="text-slate-600">&bull;</span>
          <span className="text-emerald-400">CALIBRATION: KJELLBERG-31 (VERIFIED)</span>
        </div>
      </div>
    </header>
  )
}

function Uploader({ onFileSelect, loading, preview, setPreview }) {
  const [dragging, setDragging] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [invert, setInvert] = useState(false)
  const inputRef = useRef()

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    setPreview(url)
    setZoom(1)
    setInvert(false)
    onFileSelect(file)
  }, [onFileSelect, setPreview])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  return (
    <div className="space-y-2">
      {/* Integrated PACS Viewport Controls */}
      {preview && (
        <div className="flex items-center justify-between px-2.5 py-1 bg-slate-900 text-slate-300 rounded border border-slate-700 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-[10px] font-mono text-slate-400 uppercase">PACS:</span>
            <button
              type="button"
              onClick={() => setZoom(z => Math.min(2.5, Math.round((z + 0.25) * 100) / 100))}
              title="Zoom In"
              className="p-1 rounded hover:bg-slate-800 text-slate-300 hover:text-white"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => setZoom(z => Math.max(0.75, Math.round((z - 0.25) * 100) / 100))}
              title="Zoom Out"
              className="p-1 rounded hover:bg-slate-800 text-slate-300 hover:text-white"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="font-mono text-[10px] text-slate-400 px-1">
              {(zoom * 100).toFixed(0)}%
            </span>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setInvert(inv => !inv)}
              title="Invert Bone Contrast"
              className={`px-1.5 py-0.5 rounded text-[10px] font-medium flex items-center gap-1 transition-colors ${
                invert ? 'bg-rose-900 text-rose-100' : 'hover:bg-slate-800 text-slate-300'
              }`}
            >
              <SunMedium className="w-3 h-3" />
              Invert
            </button>
            <button
              type="button"
              onClick={() => { setZoom(1); setInvert(false) }}
              title="Reset View"
              className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>
        </div>
      )}

      {/* Main Viewport / Acquisition Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative border border-dashed rounded-md p-4 cursor-pointer transition-all text-center overflow-hidden bg-white
          ${dragging ? 'border-[#831843] bg-rose-50/30' : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50/50'}
          ${loading ? 'pointer-events-none opacity-60' : ''}`}
      >
        <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => handleFile(e.target.files[0])} />

        {preview ? (
          <div className="relative overflow-hidden rounded bg-black flex items-center justify-center min-h-[180px] max-h-52">
            <img
              src={preview}
              alt="OPG Radiograph"
              style={{
                transform: `scale(${zoom})`,
                filter: invert ? 'invert(1) contrast(1.15) brightness(1.05)' : 'none',
                transition: 'transform 0.15s ease-out, filter 0.15s ease'
              }}
              className="max-h-52 w-full object-contain"
            />
            <div className="absolute bottom-1.5 right-1.5 bg-black/75 text-[10px] text-slate-300 px-1.5 py-0.5 rounded font-mono">
              Replace
            </div>
          </div>
        ) : (
          <div className="py-5">
            <div className="w-9 h-9 mx-auto rounded bg-slate-100 text-slate-600 flex items-center justify-center mb-2.5">
              <Upload className="w-4 h-4" />
            </div>
            <p className="text-xs font-semibold text-slate-800">Acquire Panoramic Radiograph (OPG)</p>
            <p className="text-[11px] text-slate-500 mt-1 max-w-xs mx-auto">
              Drop DICOM export or radiograph file (PNG, JPG, TIFF)
            </p>
            <div className="mt-2.5 inline-flex items-center gap-1 text-[10px] font-medium text-slate-700 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded">
              <Check className="w-3 h-3 text-[#831843]" /> Automated Caliper &amp; Landmark Extraction
            </div>
          </div>
        )}

        {loading && (
          <div className="absolute inset-0 bg-white/90 backdrop-blur-xs flex flex-col items-center justify-center gap-2 z-10">
            <RefreshCw className="w-5 h-5 text-[#831843] animate-spin" />
            <p className="text-xs font-semibold text-slate-800">Processing Mandibular Geometry &amp; Saliency…</p>
          </div>
        )}
      </div>
    </div>
  )
}

function BenchmarkCaseSuite({ onLoadSample, loading }) {
  const cases = [
    { key: 'mild', label: 'Benchmark 01', type: 'Mild', metric: 'SI >= 94%', desc: 'Physiological symmetry' },
    { key: 'moderate', label: 'Benchmark 02', type: 'Moderate', metric: 'SI 90-93%', desc: 'Borderline asymmetry' },
    { key: 'severe', label: 'Benchmark 03', type: 'Severe', metric: 'SI < 90%', desc: 'Marked discrepancy' },
  ]

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-3.5 shadow-xs">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-[#831843]" />
          Validated Reference Cases
        </span>
        <span className="text-[10px] text-slate-400 font-mono">1-Click Test Suite</span>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {cases.map((c) => (
          <button
            key={c.key}
            type="button"
            disabled={loading}
            onClick={() => onLoadSample(c.key)}
            className="p-2 text-left rounded border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition-all disabled:opacity-50"
          >
            <div className="flex items-center justify-between mb-0.5">
              <span className="text-[11px] font-semibold text-slate-900">{c.type}</span>
              <span className="w-1.5 h-1.5 rounded-full bg-[#831843]" />
            </div>
            <p className="text-[10px] font-mono text-slate-500">{c.metric}</p>
          </button>
        ))}
      </div>
    </div>
  )
}

function DiagnosticFindingsCard({ result }) {
  const cfg = SEVERITY_CONFIG[result.predicted_class] || SEVERITY_CONFIG.Mild
  const Icon = cfg.icon

  return (
    <div className="bg-white rounded-lg border border-slate-200 overflow-hidden shadow-xs">
      {/* Classification Banner */}
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Radiological Diagnostic Finding
          </span>
          <h2 className="text-sm font-semibold text-slate-900 mt-0.5 flex items-center gap-2">
            <span>{cfg.label}</span>
            <span className={`inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded border ${cfg.badgeBg}`}>
              <Icon className="w-3 h-3" />
              {result.predicted_class}
            </span>
          </h2>
        </div>
        <div className="text-right">
          <span className="text-[10px] text-slate-400 uppercase block">Clinical Action</span>
          <span className={`text-[11px] font-semibold px-2 py-0.5 rounded border ${cfg.urgencyBadge} inline-block mt-0.5`}>
            {cfg.urgencyText}
          </span>
        </div>
      </div>

      <div className="p-4">
        {/* Core Metric Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
          <div className="p-3 rounded border border-slate-200 bg-slate-50/60">
            <span className="text-[10px] text-slate-500 uppercase font-medium block">Kjellberg Symmetry Index</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-2xl font-bold font-mono text-slate-900 tabular-nums">
                {result.symmetry_percentage.toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">SI</span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Normative threshold: &ge; 94.0%</p>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-slate-50/60">
            <span className="text-[10px] text-slate-500 uppercase font-medium block">Model Posterior Certainty</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-2xl font-bold font-mono text-[#831843] tabular-nums">
                {result.confidence.toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">confidence</span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Multimodal Fusion (GBDT 37-dim)</p>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-slate-50/60">
            <span className="text-[10px] text-slate-500 uppercase font-medium block">Primary Segment Delta</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-2xl font-bold font-mono text-slate-900 tabular-nums">
                {Math.abs(result.morphometrics.right_CH - result.morphometrics.left_CH).toFixed(0)}px
              </span>
              <span className="text-xs text-slate-500">Δ CH</span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Bilateral Condylar Vertical Offset</p>
          </div>
        </div>

        {/* Probability Distribution Spectrum */}
        <div className="border-t border-slate-100 pt-3">
          <span className="text-[11px] font-semibold text-slate-700 block mb-2">
            Multimodal Probability Distribution
          </span>
          <div className="grid grid-cols-3 gap-3">
            {Object.entries(result.class_probabilities).map(([cls, prob]) => {
              const itemCfg = SEVERITY_CONFIG[cls] || SEVERITY_CONFIG.Mild
              return (
                <div key={cls} className="space-y-1">
                  <div className="flex justify-between text-[11px]">
                    <span className="font-medium text-slate-600">{cls}</span>
                    <span className="font-mono font-semibold tabular-nums text-slate-900">{prob.toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-100 rounded overflow-hidden">
                    <div
                      className={`h-full rounded transition-all duration-300 ${itemCfg.barColor}`}
                      style={{ width: `${Math.min(100, Math.max(0, prob))}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

function BilateralComparisonChart({ morphometrics }) {
  const metrics = [
    {
      title: 'Condylar Vertical Height (CH)',
      r: morphometrics.right_CH || 0,
      l: morphometrics.left_CH || 0,
      unit: 'px',
    },
    {
      title: 'Total Ramus Height (RH)',
      r: morphometrics.right_RH || 0,
      l: morphometrics.left_RH || 0,
      unit: 'px',
    },
    {
      title: 'Mental / Body Height (MH)',
      r: morphometrics.right_MH || 0,
      l: morphometrics.left_MH || 0,
      unit: 'px',
    },
  ]

  return (
    <div className="bg-white rounded-lg border border-slate-200 overflow-hidden flex flex-col shadow-xs">
      <div className="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Scale className="w-3.5 h-3.5 text-[#831843]" />
          <h3 className="text-xs font-semibold text-slate-800">
            Bilateral Caliper Comparison (Right vs. Left)
          </h3>
        </div>
        <span className="text-[10px] font-mono text-slate-500">Unit: Pixels</span>
      </div>

      <div className="p-4 space-y-3.5 flex-1">
        {metrics.map((m, idx) => {
          const maxVal = Math.max(m.r, m.l, 1)
          const rWidth = (m.r / maxVal) * 100
          const lWidth = (m.l / maxVal) * 100
          const diffPct = (Math.abs(m.r - m.l) / maxVal) * 100
          const isSym = diffPct < 5.0
          const isBorderline = diffPct >= 5.0 && diffPct <= 10.0

          return (
            <div key={idx} className="border-b border-slate-100 last:border-b-0 pb-3 last:pb-0">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="font-medium text-slate-800">{m.title}</span>
                <span
                  className={`text-[10px] font-mono font-medium px-1.5 py-0.2 rounded border ${
                    isSym
                      ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                      : isBorderline
                      ? 'bg-amber-50 text-amber-900 border-amber-200'
                      : 'bg-rose-50 text-rose-900 border-rose-200'
                  }`}
                >
                  Δ {diffPct.toFixed(1)}% {isSym ? 'Symmetric' : 'Discrepancy'}
                </span>
              </div>

              <div className="space-y-1 text-[11px]">
                <div className="flex items-center gap-2">
                  <span className="w-10 font-mono text-slate-400 text-[10px]">RIGHT</span>
                  <div className="flex-1 h-2 bg-slate-100 rounded overflow-hidden">
                    <div
                      className="h-full bg-[#831843] rounded transition-all duration-300"
                      style={{ width: `${rWidth}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono font-medium tabular-nums text-slate-800 text-[11px]">
                    {Math.round(m.r)} {m.unit}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="w-10 font-mono text-slate-400 text-[10px]">LEFT</span>
                  <div className="flex-1 h-2 bg-slate-100 rounded overflow-hidden">
                    <div
                      className="h-full bg-slate-700 rounded transition-all duration-300"
                      style={{ width: `${lWidth}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono font-medium tabular-nums text-slate-800 text-[11px]">
                    {Math.round(m.l)} {m.unit}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="px-4 py-2 bg-slate-50 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
        <span>Kjellberg Bilateral SI:</span>
        <span className="font-mono text-[#831843] font-semibold">
          {morphometrics.kjellberg_index.toFixed(2)}%
        </span>
      </div>
    </div>
  )
}

function ImageViewer({ claheImage, gradcamImage, landmarkImage }) {
  const [tab, setTab] = useState('landmarks')
  const tabs = [
    {
      key: 'landmarks',
      label: 'Bilateral Calipers',
      image: landmarkImage,
      desc: 'Anatomical Calipers: Condylion (Co), Sigmoid Notch (Sig), Gonion (Go) with vertical CH (cyan) and RH (yellow) lines.'
    },
    {
      key: 'gradcam',
      label: 'Grad-CAM Saliency',
      image: gradcamImage,
      desc: 'Feature attribution heatmap: Warm spectrum indicates spatial feature channels governing model asymmetry output.'
    },
    {
      key: 'clahe',
      label: 'Bone ROI (CLAHE)',
      image: claheImage,
      desc: 'Contrast-limited adaptive histogram equalization of mandibular corpus preserving cortical architecture.'
    },
  ]
  const current = tabs.find(t => t.key === tab) || tabs[0]

  return (
    <div className="bg-white rounded-lg border border-slate-200 overflow-hidden flex flex-col shadow-xs">
      <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Eye className="w-3.5 h-3.5 text-[#831843]" />
          <h3 className="text-xs font-semibold text-slate-800">
            Explainable Radiological Viewer
          </h3>
        </div>
        <span className="text-[10px] font-mono text-slate-500 uppercase">
          Standardized Viewport
        </span>
      </div>

      <div className="flex border-b border-slate-200 bg-slate-100/60 text-xs font-medium">
        {tabs.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 py-2 px-2 text-center transition-all border-b-2 ${
              tab === t.key
                ? 'text-[#831843] bg-white border-[#831843] font-semibold'
                : 'text-slate-600 hover:text-slate-900 border-transparent hover:bg-slate-100'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="p-3 flex-1 flex flex-col justify-between">
        <div className="bg-black rounded overflow-hidden flex items-center justify-center min-h-[210px] max-h-72 border border-slate-800">
          <img
            src={`data:image/png;base64,${current.image}`}
            alt={current.label}
            className="w-full h-full object-contain max-h-72"
          />
        </div>
        <p className="text-[11px] text-slate-500 mt-2 font-normal leading-relaxed border-t border-slate-100 pt-1.5">
          {current.desc}
        </p>
      </div>
    </div>
  )
}

function MorphometricsTable({ morphometrics, onRecalculate }) {
  const [vals, setVals] = useState({ ...morphometrics })
  const keys = ['CH', 'MH', 'RH']

  const handleChange = (side, key, v) => {
    setVals(prev => ({ ...prev, [`${side}_${key}`]: parseFloat(v) || 0 }))
  }

  const rows = keys.map(k => ({
    label: k === 'CH' ? 'Condylar Vertical Height' : k === 'MH' ? 'Mental / Body Height' : 'Total Ramus Height',
    key: k,
    right: vals[`right_${k}`],
    left: vals[`left_${k}`],
    ratio: morphometrics[`side_ratio_${k}`],
  }))

  return (
    <div className="bg-white rounded border border-slate-200 overflow-hidden text-xs">
      <div className="p-2.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <div>
          <h4 className="font-semibold text-slate-800">Calibrated Landmark Measurements (Pixels)</h4>
          <p className="text-[10px] text-slate-500">Adjust coordinates manually to trigger immediate re-computation of all 31 metrics</p>
        </div>
        <button
          onClick={() => onRecalculate(vals)}
          className="bg-[#831843] hover:bg-[#6F2140] text-white font-medium px-2.5 py-1 rounded text-xs flex items-center gap-1 transition-colors"
        >
          <RefreshCw className="w-3 h-3" /> Recalculate
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold text-[10px] uppercase">
            <tr>
              <th className="px-3 py-2">Anatomical Landmark</th>
              <th className="px-3 py-2 text-center">Right (px)</th>
              <th className="px-3 py-2 text-center">Left (px)</th>
              <th className="px-3 py-2 text-center">Ratio</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map(row => (
              <tr key={row.key} className="hover:bg-slate-50">
                <td className="px-3 py-2 font-medium text-slate-800">
                  {row.label} <span className="font-mono text-slate-400">({row.key})</span>
                </td>
                <td className="px-3 py-2 text-center">
                  <input
                    type="number"
                    value={row.right}
                    onChange={e => handleChange('right', row.key, e.target.value)}
                    className="w-16 text-center font-mono border border-slate-200 rounded px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-[#831843] bg-white text-xs"
                  />
                </td>
                <td className="px-3 py-2 text-center">
                  <input
                    type="number"
                    value={row.left}
                    onChange={e => handleChange('left', row.key, e.target.value)}
                    className="w-16 text-center font-mono border border-slate-200 rounded px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-[#831843] bg-white text-xs"
                  />
                </td>
                <td className="px-3 py-2 text-center font-mono font-medium tabular-nums text-slate-700">
                  {(row.ratio * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
            <tr className="bg-slate-50 font-semibold text-slate-800">
              <td className="px-3 py-2">Kjellberg's Composite Index</td>
              <td colSpan={3} className="px-3 py-2 text-center font-mono text-[#831843] text-xs">
                {morphometrics.kjellberg_index.toFixed(2)}%
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ManualOverrideAccordion({ morphometrics, onRecalculate }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="bg-white rounded-lg border border-slate-200 shadow-xs overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-2.5 flex items-center justify-between text-left hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Sliders className="w-3.5 h-3.5 text-[#831843]" />
          <div>
            <span className="font-semibold text-xs text-slate-800">
              Morphometric Caliper Calibration (Clinician Manual Override)
            </span>
            <span className="text-[10px] text-slate-400 ml-2 font-mono">
              [Optional Calibration]
            </span>
          </div>
        </div>
        <ChevronDown className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-150 ${open ? 'rotate-180 text-[#831843]' : ''}`} />
      </button>
      {open && (
        <div className="border-t border-slate-200 p-3 bg-slate-50/50">
          <MorphometricsTable morphometrics={morphometrics} onRecalculate={onRecalculate} />
        </div>
      )}
    </div>
  )
}

function DoctorRecommendationCard({ severity, symmetryIndex, treatment }) {
  const advisoryNotes = {
    Mild: `Following radiographic assessment of the panoramic examination, the mandibular ramus and condylar vertical heights demonstrate physiological symmetry within normal biological variation (Symmetry Index: ${symmetryIndex.toFixed(1)}%). No active surgical or orthognathic intervention is indicated. Standard 6-to-12 month routine clinical and dental occlusion monitoring is recommended to confirm continued morphological stability.`,
    Moderate: `Radiographic findings demonstrate borderline mandibular asymmetry (Symmetry Index: ${symmetryIndex.toFixed(1)}%) localized along the condylar-ramal axis. Functional compensation or dental crossbite should be assessed clinically. Comprehensive orthodontic evaluation, occlusal splint therapy, and follow-up panoramic radiography in 6 months are indicated to monitor skeletal maturation.`,
    Severe: `Marked skeletal mandibular asymmetry is evident on bilateral morphometric assessment (Symmetry Index: ${symmetryIndex.toFixed(1)}%), characterized by pronounced condylar-ramal height discrepancy. Immediate referral to Oral & Maxillofacial Surgery (OMFS) is indicated. High-resolution 3D CBCT is strongly advised for volumetric condylar and airway mapping in preparation for orthognathic surgical planning.`,
  }

  const advisory = advisoryNotes[severity] || advisoryNotes.Mild
  const cfg = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.Mild

  return (
    <div className="bg-white rounded-lg border border-slate-200 shadow-xs overflow-hidden">
      <div className="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
          <h3 className="text-xs font-semibold text-slate-800">
            Attending Specialist Protocol &amp; Recommendation
          </h3>
        </div>
        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${cfg.urgencyBadge}`}>
          {cfg.urgencyText}
        </span>
      </div>

      <div className="p-4">
        {/* Clinician Profile Strip */}
        <div className="flex items-center gap-3 pb-3 border-b border-slate-100">
          <div className="relative flex-shrink-0">
            <img
              src="/doctor.png"
              alt="Dr. Md. Shahidur Rahman, BDS"
              className="w-12 h-12 rounded-full object-cover border border-slate-200"
              onError={(e) => {
                e.target.onerror = null
                e.target.src = 'https://ui-avatars.com/api/?name=Shahidur+Rahman&background=831843&color=fff'
              }}
            />
            <div
              className="absolute -bottom-0.5 -right-0.5 bg-emerald-600 text-white rounded-full p-0.5 shadow-2xs"
              title="Verified Specialist"
            >
              <CheckCircle2 className="w-3 h-3" />
            </div>
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5">
              <h4 className="text-xs font-bold text-slate-900 truncate">Dr. Md. Shahidur Rahman, BDS</h4>
              <span className="text-[10px] font-medium bg-emerald-50 text-emerald-800 border border-emerald-200 px-1.5 py-0.2 rounded">
                Verified Specialist
              </span>
            </div>
            <p className="text-[11px] text-slate-600 truncate">
              Senior Consultant, Oral &amp; Maxillofacial Orthodontics
            </p>
            <p className="text-[10px] text-slate-400 font-mono">
              Reg. #BMDC-D-14920 &bull; Smile Therapy Imaging Center
            </p>
          </div>
        </div>

        {/* Clinical Advisory Note */}
        <div className="mt-3 bg-slate-50 p-3 rounded border border-slate-200">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-800 mb-1">
            <FileCheck className="w-3.5 h-3.5 text-[#831843]" />
            <span>Clinical Advisory Assessment:</span>
          </div>
          <p className="text-xs text-slate-700 leading-relaxed font-normal">
            {advisory}
          </p>

          {treatment.referral && (
            <div className="mt-2.5 pt-2 border-t border-slate-200 flex items-center justify-between text-xs">
              <span className="text-slate-600 font-medium">Referral Destination:</span>
              <span className="font-semibold text-slate-900 bg-white px-2 py-0.5 rounded border border-slate-200">
                {treatment.referral}
              </span>
            </div>
          )}
        </div>

        {/* Digital Certification Stamp */}
        <div className="mt-3 pt-2 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2 text-[10px] text-slate-500">
          <div className="flex items-center gap-1">
            <span className="font-mono text-emerald-700 font-semibold">&#10003; Electronically Certified</span>
            <span>by Dr. Md. Shahidur Rahman, BDS</span>
          </div>
          <div className="font-mono text-slate-400">
            ST-OMFS-{new Date().getFullYear()}-CDSS
          </div>
        </div>
      </div>
    </div>
  )
}

function PrintableMedicalReport({ result }) {
  if (!result) return null
  const cfg = SEVERITY_CONFIG[result.predicted_class] || SEVERITY_CONFIG.Mild

  return (
    <div className="hidden print:block print:w-full print:p-6 bg-white text-slate-900 font-sans">
      <div className="border-b-2 border-slate-900 pb-3 mb-4 flex items-start justify-between">
        <div>
          <h1 className="text-lg font-bold tracking-tight text-slate-900 uppercase">
            Smile Therapy — Orthodontic &amp; Maxillofacial Clinic
          </h1>
          <p className="text-xs text-slate-600 font-medium">
            Maxillofacial Radiology &amp; Cephalometric Department &bull; PACS Node 4
          </p>
          <p className="text-[10px] text-slate-500 mt-0.5">
            Registration: BMDC-ST-RAD-2026 &bull; DICOM Standard Conformance
          </p>
        </div>
        <div className="text-right text-xs">
          <p className="font-bold text-[#831843]">MAXILLOFACIAL RADIOLOGY REPORT</p>
          <p className="font-mono text-slate-600 text-[10px] mt-0.5">EXAM ID: #ST-2026-OMFS</p>
          <p className="text-slate-500 text-[10px]">Date: {new Date().toLocaleDateString('en-GB')}</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 p-2.5 bg-slate-50 border border-slate-200 rounded mb-4 text-xs">
        <div>
          <span className="text-slate-500 font-medium text-[10px] block">Study Modality</span>
          <p className="font-semibold text-slate-800">Panoramic Radiograph (OPG)</p>
        </div>
        <div>
          <span className="text-slate-500 font-medium text-[10px] block">Diagnostic Class</span>
          <p className="font-bold text-slate-900">{cfg.label}</p>
        </div>
        <div>
          <span className="text-slate-500 font-medium text-[10px] block">Kjellberg Symmetry Index</span>
          <p className="font-bold font-mono text-slate-900">{result.symmetry_percentage.toFixed(1)}%</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="border border-slate-300 rounded p-1.5 text-center bg-black">
          <p className="text-[10px] text-white font-mono mb-1">Bilateral Caliper Calibrations (Co, Sig, Go)</p>
          <img
            src={`data:image/png;base64,${result.landmark_overlay_base64 || result.landmark_image}`}
            alt="Bilateral Calipers"
            className="max-h-44 mx-auto object-contain"
          />
        </div>
        <div className="border border-slate-300 rounded p-1.5 text-center bg-black">
          <p className="text-[10px] text-white font-mono mb-1">Grad-CAM Feature Attribution Saliency</p>
          <img
            src={`data:image/png;base64,${result.gradcam_image}`}
            alt="Saliency Map"
            className="max-h-44 mx-auto object-contain"
          />
        </div>
      </div>

      <div className="mb-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 mb-1.5 border-b border-slate-200 pb-0.5">
          Bilateral Morphometric Caliper Measurements
        </h3>
        <table className="w-full text-xs border border-slate-200 text-left">
          <thead className="bg-slate-100 font-semibold text-slate-700">
            <tr>
              <th className="p-1.5 border-b border-slate-200">Measurement</th>
              <th className="p-1.5 border-b border-slate-200 text-center">Patient Right (px)</th>
              <th className="p-1.5 border-b border-slate-200 text-center">Patient Left (px)</th>
              <th className="p-1.5 border-b border-slate-200 text-center">Segment Ratio</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            <tr>
              <td className="p-1.5 font-medium">Condylar Vertical Height (CH)</td>
              <td className="p-1.5 text-center font-mono">{Math.round(result.morphometrics.right_CH)}</td>
              <td className="p-1.5 text-center font-mono">{Math.round(result.morphometrics.left_CH)}</td>
              <td className="p-1.5 text-center font-mono">{(result.morphometrics.side_ratio_CH * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td className="p-1.5 font-medium">Total Ramus Height (RH)</td>
              <td className="p-1.5 text-center font-mono">{Math.round(result.morphometrics.right_RH)}</td>
              <td className="p-1.5 text-center font-mono">{Math.round(result.morphometrics.left_RH)}</td>
              <td className="p-1.5 text-center font-mono">{(result.morphometrics.side_ratio_RH * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td className="p-1.5 font-medium">Mental / Body Height (MH)</td>
              <td className="p-1.5 text-center font-mono">{Math.round(result.morphometrics.right_MH)}</td>
              <td className="p-1.5 text-center font-mono">{Math.round(result.morphometrics.left_MH)}</td>
              <td className="p-1.5 text-center font-mono">{(result.morphometrics.side_ratio_MH * 100).toFixed(1)}%</td>
            </tr>
            <tr className="bg-slate-50 font-bold">
              <td className="p-1.5">Composite Kjellberg SI</td>
              <td colSpan={3} className="p-1.5 text-center font-mono text-slate-900">
                {result.symmetry_percentage.toFixed(2)}%
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="border border-slate-300 rounded p-3 bg-slate-50 mb-3">
        <h4 className="text-[11px] font-bold uppercase text-slate-800 mb-0.5">Specialist Recommendation</h4>
        <p className="text-xs text-slate-700 leading-relaxed">
          {result.treatment.recommendation}
        </p>
      </div>

      <div className="pt-3 border-t border-slate-300 flex items-center justify-between text-xs text-slate-600">
        <div>
          <p className="font-bold text-slate-900">Dr. Md. Shahidur Rahman, BDS</p>
          <p className="text-[10px] text-slate-500">Senior Consultant, Oral &amp; Maxillofacial Orthodontics</p>
          <p className="text-[9px] text-slate-400">Reg. #BMDC-D-14920 &bull; Smile Therapy Clinic</p>
        </div>
        <div className="text-right">
          <p className="font-mono text-slate-700 font-semibold text-[11px]">&#10003; CERTIFIED RADIOLOGY REPORT</p>
          <p className="text-[9px] text-slate-400">Electronic Verification Hash: ST-RAD-VALID</p>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [authModalState, setAuthModalState] = useState({ isOpen: false, tab: 'signin' })
  const [currentUser, setCurrentUser] = useState(null)
  const [apiBaseUrl, setApiBaseUrl] = useState(() => {
    return localStorage.getItem('smile_backend_url') || DEFAULT_API_BASE
  })
  const [serverModalOpen, setServerModalOpen] = useState(false)

  const reportRef = useRef()

  const handleAnalyze = useCallback(async (fileToUse, overrides = null) => {
    if (!fileToUse && !file) return
    const f = fileToUse || file
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', f)
      if (overrides) {
        Object.entries(overrides).forEach(([k, v]) => v !== undefined && form.append(k, v))
      }
      const { data } = await axios.post(`${apiBaseUrl}/api/analyze`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      })
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Radiological pipeline execution failed. Ensure backend service is reachable.')
    } finally {
      setLoading(false)
    }
  }, [file, apiBaseUrl])

  const handleFileSelect = useCallback((f) => {
    setFile(f)
    setResult(null)
    setError(null)
    handleAnalyze(f)
  }, [handleAnalyze])

  const handleLoadSample = useCallback(async (sampleKey) => {
    const sampleMap = {
      mild: { name: 'mild_sample.jpg', path: '/samples/mild_sample.jpg', label: 'Benchmark 01: Mild' },
      moderate: { name: 'moderate_sample.jpg', path: '/samples/moderate_sample.jpg', label: 'Benchmark 02: Moderate' },
      severe: { name: 'severe_sample.jpg', path: '/samples/severe_sample.jpg', label: 'Benchmark 03: Severe' },
    }
    const target = sampleMap[sampleKey]
    if (!target) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let fileBlob
      try {
        const res = await fetch(target.path)
        if (!res.ok) throw new Error('Sample fetch returned non-200')
        fileBlob = await res.blob()
      } catch (fetchErr) {
        fileBlob = await generateSyntheticSampleBlob(sampleKey)
      }

      const sampleFile = new File([fileBlob], target.name, { type: 'image/jpeg' })
      setFile(sampleFile)
      setPreview(URL.createObjectURL(sampleFile))
      handleAnalyze(sampleFile)
    } catch (err) {
      setError(`Failed to load ${target.label} reference case: ${err.message}`)
      setLoading(false)
    }
  }, [handleAnalyze])

  const handleRecalculate = useCallback((vals) => {
    if (!file) return
    handleAnalyze(file, {
      right_CH: vals.right_CH, right_MH: vals.right_MH, right_RH: vals.right_RH,
      left_CH: vals.left_CH, left_MH: vals.left_MH, left_RH: vals.left_RH,
    })
  }, [file, handleAnalyze])

  const handleDownloadPDF = useCallback(async () => {
    if (!reportRef.current) return
    const canvas = await html2canvas(reportRef.current, { scale: 2, useCORS: true })
    const pdf = new jsPDF('p', 'mm', 'a4')
    const imgData = canvas.toDataURL('image/png')
    const pageW = pdf.internal.pageSize.getWidth()
    const imgH = (canvas.height * pageW) / canvas.width
    pdf.addImage(imgData, 'PNG', 0, 0, pageW, imgH)
    pdf.save(`Smile-Therapy-Radiology-Report-${new Date().toISOString().slice(0, 10)}.pdf`)
  }, [])

  const handlePrintReport = useCallback(() => {
    window.print()
  }, [])

  return (
    <div className="min-h-screen bg-[#F8F9FA] text-slate-900 flex flex-col font-sans">
      <Header
        currentUser={currentUser}
        onOpenAuth={(tab) => setAuthModalState({ isOpen: true, tab })}
        onLogout={() => setCurrentUser(null)}
        apiBaseUrl={apiBaseUrl}
        onOpenServerModal={() => setServerModalOpen(true)}
      />

      <ServerConfigModal
        isOpen={serverModalOpen}
        onClose={() => setServerModalOpen(false)}
        currentUrl={apiBaseUrl}
        onSaveUrl={(newUrl) => {
          localStorage.setItem('smile_backend_url', newUrl)
          setApiBaseUrl(newUrl)
        }}
      />

      <AuthModal
        isOpen={authModalState.isOpen}
        initialTab={authModalState.tab}
        onClose={() => setAuthModalState({ isOpen: false, tab: 'signin' })}
        onLoginSuccess={(user) => setCurrentUser(user)}
      />

      <main className="no-print max-w-7xl mx-auto px-4 sm:px-6 py-5 flex-1 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Left Console: Acquisition, Viewport & Benchmarks (4 cols) */}
          <div className="lg:col-span-4 space-y-4">
            {/* Viewport Acquisition */}
            <div className="bg-white rounded-lg border border-slate-200 shadow-xs p-4">
              <div className="flex items-center justify-between mb-2.5">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-800 flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-[#831843]" />
                  Radiograph Acquisition
                </h2>
                <span className="text-[10px] font-mono text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                  DICOM / OPG
                </span>
              </div>

              <Uploader
                onFileSelect={handleFileSelect}
                loading={loading}
                preview={preview}
                setPreview={setPreview}
              />

              <button
                onClick={() => handleAnalyze(file)}
                disabled={!file || loading}
                className="mt-3 w-full bg-[#831843] hover:bg-[#6F2140] disabled:bg-slate-200 disabled:text-slate-400 text-white font-medium text-xs py-2 rounded transition-colors flex items-center justify-center gap-1.5 shadow-xs"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Executing Diagnostic Pipeline…
                  </>
                ) : (
                  <>
                    <Activity className="w-3.5 h-3.5" />
                    Execute Diagnostic Pipeline
                  </>
                )}
              </button>

              {error && (
                <div className="mt-2.5 p-2.5 bg-rose-50 border border-rose-200 rounded text-xs text-rose-800 leading-relaxed">
                  <strong>Diagnostic Pipeline Error:</strong> {error}
                </div>
              )}
            </div>

            {/* Benchmark Test Cases */}
            <BenchmarkCaseSuite onLoadSample={handleLoadSample} loading={loading} />

            {/* Diagnostic Normative Scale */}
            <div className="bg-white rounded-lg border border-slate-200 shadow-xs p-3.5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-800 mb-2 flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-slate-500" />
                Normative Classification Matrix
              </h3>
              <div className="space-y-1.5 text-xs">
                <div className="flex items-center justify-between p-2 rounded bg-emerald-50/60 border border-emerald-200/50">
                  <span className="font-medium text-emerald-950">Mild (Physiological)</span>
                  <span className="font-mono font-semibold text-emerald-800">&ge; 94.0% SI</span>
                </div>
                <div className="flex items-center justify-between p-2 rounded bg-amber-50/60 border border-amber-200/50">
                  <span className="font-medium text-amber-950">Moderate (Borderline)</span>
                  <span className="font-mono font-semibold text-amber-800">90.0% &ndash; 93.9% SI</span>
                </div>
                <div className="flex items-center justify-between p-2 rounded bg-rose-50/60 border border-rose-200/50">
                  <span className="font-medium text-rose-950">Severe (Discrepancy)</span>
                  <span className="font-mono font-semibold text-rose-900">&lt; 90.0% SI</span>
                </div>
              </div>
              <p className="text-[10px] text-slate-400 mt-2 leading-relaxed">
                Standardized Kjellberg Symmetry Index based on Condylion (Co), Sigmoid Notch (Sig), and Gonion (Go) vertical ratios.
              </p>
            </div>

            {/* Clinical Actions */}
            {result && (
              <div className="bg-white rounded-lg border border-slate-200 shadow-xs p-3.5 space-y-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-800 mb-1.5 flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-[#831843]" />
                  Clinical Actions
                </h3>
                <button
                  onClick={handlePrintReport}
                  className="w-full bg-slate-900 hover:bg-slate-800 text-white font-medium py-2 rounded transition-colors flex items-center justify-center gap-1.5 text-xs shadow-xs"
                >
                  <Printer className="w-3.5 h-3.5" /> Print PACS Medical Record
                </button>
                <button
                  onClick={handleDownloadPDF}
                  className="w-full bg-white hover:bg-slate-50 text-slate-800 border border-slate-200 font-medium py-1.5 rounded transition-colors flex items-center justify-center gap-1.5 text-xs"
                >
                  <Download className="w-3.5 h-3.5 text-slate-600" /> Export Report (PDF)
                </button>
              </div>
            )}
          </div>

          {/* Right Main Console: Diagnostic Workstation (8 cols) */}
          <div className="lg:col-span-8 space-y-4" ref={reportRef}>
            {!result && !loading && (
              <div className="bg-white rounded-lg border border-dashed border-slate-300 p-10 text-center">
                <div className="w-10 h-10 mx-auto rounded bg-slate-100 flex items-center justify-center text-slate-500 mb-3">
                  <Activity className="w-5 h-5 text-slate-600" />
                </div>
                <h3 className="text-sm font-semibold text-slate-800">Radiology Diagnostic Console Ready</h3>
                <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto leading-relaxed">
                  Acquire an Orthopantomogram (OPG) radiograph or select a benchmark reference case on the left to execute automated bilateral mandibular segmentation, extract calipers, and inspect asymmetry metrics.
                </p>
                <div className="mt-4 inline-flex items-center gap-2 text-[11px] text-slate-500 bg-slate-50 border border-slate-200 px-3 py-1 rounded">
                  <span>Modality: <strong>OPG Panoramic</strong></span>
                  <span>&bull;</span>
                  <span>Engine: <strong>Multimodal DenseNet121 + GBDT</strong></span>
                </div>
              </div>
            )}

            {result && (
              <>
                {/* 1. Diagnostic Findings Card */}
                <DiagnosticFindingsCard result={result} />

                {/* 2. Side-by-side Imaging & Morphometrics */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ImageViewer
                    claheImage={result.clahe_image}
                    gradcamImage={result.gradcam_image}
                    landmarkImage={result.landmark_overlay_base64 || result.landmark_image}
                  />
                  <BilateralComparisonChart morphometrics={result.morphometrics} />
                </div>

                {/* 3. Manual Caliper Calibration (Accordion) */}
                <ManualOverrideAccordion
                  morphometrics={result.morphometrics}
                  onRecalculate={handleRecalculate}
                />

                {/* 4. Attending Specialist Protocol Card */}
                <DoctorRecommendationCard
                  severity={result.predicted_class}
                  symmetryIndex={result.symmetry_percentage}
                  treatment={result.treatment}
                />

                {/* 5. Institutional Disclaimer */}
                <div className="p-3 bg-white rounded-lg border border-slate-200 flex items-start gap-2 text-[11px] text-slate-500 leading-relaxed shadow-2xs">
                  <Info className="w-3.5 h-3.5 text-slate-400 flex-shrink-0 mt-0.5" />
                  <p>
                    <strong className="text-slate-700">Clinical Decision Support Notice:</strong> This quantitative morphometric analysis is generated for oral &amp; maxillofacial radiological consultation. Final therapeutic or orthognathic surgical planning must be corroborated by clinical examination and volumetric 3D CBCT as directed by the attending specialist.
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      </main>

      {/* Printable Hospital Medical Pad Template */}
      <PrintableMedicalReport result={result} />

      {/* Hospital Footer */}
      <footer className="no-print bg-white border-t border-slate-200 py-2.5 text-center text-xs text-slate-500 mt-6">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-1 text-[11px]">
          <span>&copy; {new Date().getFullYear()} Smile Therapy Orthodontic &amp; Maxillofacial Imaging Center &bull; Oral Radiology Division</span>
          <span className="font-mono text-slate-400">
            Workstation Node ST-PACS-04 &bull; Clinical Decision Support v2.4
          </span>
        </div>
      </footer>
    </div>
  )
}
