"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const vscode = require("vscode");
const getArgs_1 = require("./getArgs");
function activate(context) {
    let disposable = vscode.commands.registerCommand('extension.debugFlags', getArgs_1.getArgs);
    context.subscriptions.push(disposable);
}
exports.activate = activate;
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map