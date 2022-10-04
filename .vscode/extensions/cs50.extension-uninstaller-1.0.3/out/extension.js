"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = require("vscode");
/**
 * Uninstall extensions by their extension identifiers specified in settings.json.
 *
 * @param context vscode.ExtensionContext
 */
function uninstallExtension(context) {
    // Get the extension name for current context
    const extName = context.extension.packageJSON['name'];
    try {
        // Skip uninstalling extensions specified in the configuration
        const skip = vscode.workspace.getConfiguration(extName)['skip'];
        // Get the configuration for this extension
        vscode.workspace.getConfiguration(extName)['uninstall'].map((each) => {
            // Remove matching extension
            if (vscode.extensions.getExtension(each) !== undefined) {
                if (!skip.includes(each)) {
                    vscode.commands.executeCommand('workbench.extensions.uninstallExtension', each);
                }
            }
        });
    }
    catch (e) {
        console.error(e);
    }
}
function activate(context) {
    uninstallExtension(context);
    vscode.extensions.onDidChange(() => { uninstallExtension(context); });
    vscode.workspace.onDidChangeConfiguration(() => { uninstallExtension(context); });
}
exports.activate = activate;
function deactivate() {
    // Not used.
}
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map