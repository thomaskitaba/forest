"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = require("vscode");
function activate(context) {
    const provider = new DDBViewProvider(context.extensionUri);
    context.subscriptions.push(vscode.window.registerWebviewViewProvider(DDBViewProvider.viewId, provider));
}
exports.activate = activate;
class DDBViewProvider {
    constructor(_extensionUri) {
        this._extensionUri = _extensionUri;
    }
    resolveWebviewView(webviewView, _context, _token) {
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
    }
    _getHtmlForWebview(webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'static', 'ddb.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'static', 'style.css'));
        return `
			<!DOCTYPE html>
			<html lang="en">
				<head>
					<meta charset="UTF-8">
					<meta name="viewport" content="initial-scale=1.0, width=device-width">
					<link href="${styleUri}" rel="stylesheet">
					<script src="${scriptUri}"></script>
					<title>ddb50</title>
				</head>
				<body>
					<div id="ddbChatContainer">
						<div id="ddbChatText"></div>
						<div id="ddbInput"><textarea placeholder="Message ddb"></textarea></div>
					</div>
				</body>
			</html>
		`;
    }
}
DDBViewProvider.viewId = 'ddb50.debugView';
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map