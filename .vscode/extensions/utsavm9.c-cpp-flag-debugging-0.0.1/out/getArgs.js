"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const vscode = require("vscode");
function getArgs() {
    return __awaiter(this, void 0, void 0, function* () {
        //Replacing the args in launch.json
        vscode.workspace.findFiles('.vscode/launch.json').then((files) => __awaiter(this, void 0, void 0, function* () {
            //Ensuring the file exists beforehand
            if (files.length < 1) {
                vscode.window.showWarningMessage('A launch.json does not already exist. Create one now or use "Debug: Open launch.json" in the command palette later. Then try again to debug with arguments.');
                vscode.commands.executeCommand('debug.addConfiguration');
                return;
            }
            //Getting flags
            const quickPicked = yield vscode.window.showInputBox({
                prompt: "Enter flags to debug your program with"
            });
            //Converting to array of arguments
            let args;
            if (quickPicked) {
                args = quickPicked.split(' ');
            }
            else {
                return;
            }
            //Processing each argument for quotes
            for (let index in args) {
                const quote = /\"/g;
                args[index] = `"${args[index].replace(quote, '\\\\\\"')}"`;
            }
            const argsJSON = args.join(", ");
            //Replacing in launch.json
            vscode.workspace.openTextDocument(files[0]).then((launchJSON) => __awaiter(this, void 0, void 0, function* () {
                //Saving the state of file for restore.
                if (launchJSON.isDirty) {
                    yield launchJSON.save();
                }
                //Removing all current args property and adding new ones
                const args = /"args"\s*:\s*\[.*\]\s*,/g;
                const nameProp = /(([\t\ ]*)"name"\s*:)/g;
                let argsDoc = launchJSON.getText().replace(args, '').replace(nameProp, `$2"args": [${argsJSON}],\n$1`);
                let wEdit = new vscode.WorkspaceEdit();
                wEdit.replace(launchJSON.uri, launchJSON.validateRange(new vscode.Range(0, 0, Infinity, Infinity)), argsDoc);
                //Applying the edit
                vscode.workspace.applyEdit(wEdit).then(() => __awaiter(this, void 0, void 0, function* () {
                    let cppTools = vscode.extensions.getExtension('ms-vscode.cpptools');
                    if (cppTools && cppTools.isActive === false) {
                        cppTools.activate().then(() => __awaiter(this, void 0, void 0, function* () {
                            yield vscode.commands.executeCommand('C_Cpp.BuildAndDebugActiveFile').then();
                        }), reason => {
                            vscode.window.showInformationMessage("Unable to launch Microsoft C/C++ Debug Tools extension. Is it installed?", reason);
                        });
                    }
                    else {
                        yield vscode.commands.executeCommand('C_Cpp.BuildAndDebugActiveFile');
                    }
                }));
            }), reason => {
                vscode.window.showInformationMessage("Unable to open launch.json.", reason);
            });
        }), reason => {
            vscode.window.showInformationMessage("Unable to search for launch.json.", reason);
        });
    });
}
exports.getArgs = getArgs;
//# sourceMappingURL=getArgs.js.map