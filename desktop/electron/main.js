"use strict";

const { app, BrowserWindow, Menu, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const http = require("http");
const net = require("net");
const path = require("path");

const HOST = "127.0.0.1";
const SERVER_READY_TIMEOUT_MS = 30000;

let mainWindow = null;
let dashboardProcess = null;
let dashboardUrl = null;
let isQuitting = false;
let serverOutput = "";
const desktopToken = crypto.randomBytes(24).toString("hex");

function appendServerOutput(chunk) {
  serverOutput = `${serverOutput}${chunk.toString()}`.slice(-4000);
}

function resolveProjectRoot() {
  const candidates = [
    process.env.CRYPTO_AI_TRADER_ROOT,
    path.resolve(path.dirname(process.execPath), "..", ".."),
    path.resolve(__dirname, "..", ".."),
    process.resourcesPath,
    process.cwd(),
    path.dirname(process.execPath),
  ].filter(Boolean);

  const attempted = [];
  for (const candidate of candidates) {
    const projectRoot = path.resolve(candidate);
    attempted.push(projectRoot);
    if (fs.existsSync(resolvePythonPath(projectRoot))) {
      return { projectRoot, attempted };
    }
  }

  return { projectRoot: path.resolve(candidates[0] || __dirname), attempted };
}

function resolvePythonPath(projectRoot) {
  if (process.platform === "win32") {
    return path.join(projectRoot, ".venv", "Scripts", "python.exe");
  }

  return path.join(projectRoot, ".venv", "bin", "python");
}

function findFreePort(host) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, host, () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : null;
      server.close(() => {
        if (port) {
          resolve(port);
        } else {
          reject(new Error("Unable to resolve a free local port."));
        }
      });
    });
  });
}

function waitForServer(url, timeoutMs) {
  const startedAt = Date.now();
  let settled = false;

  return new Promise((resolve, reject) => {
    const poll = () => {
      if (settled) {
        return;
      }

      const request = http.get(url, (response) => {
        response.resume();
        if (response.statusCode && response.statusCode < 500) {
          settled = true;
          resolve();
          return;
        }
        retry();
      });

      request.on("error", retry);
      request.setTimeout(1000, () => {
        request.destroy();
        retry();
      });
    };

    const retry = () => {
      if (settled) {
        return;
      }

      if (Date.now() - startedAt > timeoutMs) {
        settled = true;
        reject(new Error(`Dashboard server did not become ready within ${timeoutMs}ms.`));
        return;
      }
      setTimeout(poll, 250);
    };

    poll();
  });
}

function formatLaunchError(error, context) {
  const lines = [
    error.stack || error.message || String(error),
    "",
    `Project root: ${context.projectRoot}`,
    `Python: ${context.pythonPath}`,
    `Command: ${context.pythonPath} ${context.args.join(" ")}`,
    "Tried roots:",
    ...context.attempted.map((item) => `- ${item}`),
  ];

  const output = serverOutput.trim();
  if (output) {
    lines.push("", "Recent server output:", output);
  }

  return lines.join("\n");
}

function setupMenu(projectRoot) {
  const reportsPath = path.join(projectRoot, "reports");
  const template = [
    {
      label: "Crypto AI Trader",
      submenu: [
        {
          label: "刷新控制台",
          accelerator: "F5",
          click: () => mainWindow?.reload(),
        },
        {
          label: "打开报告目录",
          click: () => shell.openPath(reportsPath),
        },
        {
          label: "打开项目目录",
          click: () => shell.openPath(projectRoot),
        },
        { type: "separator" },
        {
          label: "开发者工具",
          accelerator: "Ctrl+Shift+I",
          click: () => mainWindow?.webContents.openDevTools({ mode: "detach" }),
        },
        { type: "separator" },
        {
          label: "退出",
          role: "quit",
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

async function startDashboardServer() {
  const { projectRoot, attempted } = resolveProjectRoot();
  const pythonPath = resolvePythonPath(projectRoot);

  if (!fs.existsSync(pythonPath)) {
    throw new Error(`Python venv not found: ${pythonPath}\n\nTried roots:\n${attempted.join("\n")}`);
  }

  const port = await findFreePort(HOST);
  const args = [
    "-m",
    "crypto_ai_trader.dashboard_server",
    "--host",
    HOST,
    "--port",
    String(port),
  ];

  dashboardUrl = `http://${HOST}:${port}/`;
  let startupComplete = false;
  dashboardProcess = spawn(pythonPath, args, {
    cwd: projectRoot,
    env: {
      ...process.env,
      CRYPTO_AI_TRADER_ROOT: projectRoot,
      CRYPTO_AI_DESKTOP_TOKEN: desktopToken,
      PYTHONUTF8: "1",
      PYTHONIOENCODING: "utf-8",
      PYTHONUNBUFFERED: "1",
    },
    shell: false,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  dashboardProcess.stdout.on("data", appendServerOutput);
  dashboardProcess.stderr.on("data", appendServerOutput);

  dashboardProcess.once("error", (error) => {
    appendServerOutput(error.stack || error.message || String(error));
  });

  dashboardProcess.once("exit", (code, signal) => {
    dashboardProcess = null;
    if (!startupComplete) {
      return;
    }
    if (!isQuitting) {
      const reason = signal || `exit code ${code}`;
      dialog.showErrorBox(
        "Dashboard server stopped",
        `The local dashboard server stopped unexpectedly (${reason}).\n\n${serverOutput.trim()}`
      );
      app.quit();
    }
  });

  const context = { projectRoot, pythonPath, args, attempted };
  const processFailed = new Promise((_, reject) => {
    dashboardProcess.once("error", (error) => {
      reject(new Error(`Dashboard server process failed to start: ${error.message}`));
    });
    dashboardProcess.once("exit", (code, signal) => {
      reject(new Error(`Dashboard server exited before it was ready (${signal || `exit code ${code}`}).`));
    });
  });

  try {
    await Promise.race([waitForServer(dashboardUrl, SERVER_READY_TIMEOUT_MS), processFailed]);
    startupComplete = true;
    return { url: dashboardUrl, projectRoot };
  } catch (error) {
    throw new Error(formatLaunchError(error, context));
  }
}

function stopDashboardServer() {
  isQuitting = true;

  if (!dashboardProcess || dashboardProcess.killed) {
    dashboardProcess = null;
    return;
  }

  dashboardProcess.removeAllListeners("exit");
  dashboardProcess.kill();
  dashboardProcess = null;
}

async function createWindow() {
  const { url, projectRoot } = await startDashboardServer();
  setupMenu(projectRoot);

  mainWindow = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 960,
    minHeight: 640,
    title: "Crypto AI Trader",
    backgroundColor: "#0f172a",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url: nextUrl }) => {
    try {
      const parsed = new URL(nextUrl);
      if (["http:", "https:"].includes(parsed.protocol)) {
        shell.openExternal(nextUrl);
      }
    } catch {
      return { action: "deny" };
    }
    return { action: "deny" };
  });
  mainWindow.webContents.session.webRequest.onBeforeSendHeaders((details, callback) => {
    if (dashboardUrl && details.url.startsWith(dashboardUrl)) {
      details.requestHeaders["X-Desktop-Token"] = desktopToken;
    }
    callback({ requestHeaders: details.requestHeaders });
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
    stopDashboardServer();
  });

  await mainWindow.loadURL(url);
}

const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!mainWindow) {
      return;
    }
    if (mainWindow.isMinimized()) {
      mainWindow.restore();
    }
    mainWindow.focus();
  });

  app.whenReady().then(createWindow).catch((error) => {
    dialog.showErrorBox("Unable to start desktop dashboard", error.stack || error.message);
    stopDashboardServer();
    app.quit();
  });
}

app.on("before-quit", () => {
  stopDashboardServer();
});

app.on("window-all-closed", () => {
  stopDashboardServer();
  app.quit();
});
