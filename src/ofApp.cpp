#include "ofApp.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ofxIni/ofxIniFile.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "States/MenuState.h"
#include "Keyboard/KeyboardInput.h"
#include "ofxTimer/ofxTimer.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofxIniFile settings("settings.ini");
	engine = ofxGameEngine::Instance();
	ofSetEscapeQuitsApp(false);


	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------
	/* Load the games default settings and apply them to the game. If the settings
	do not exist, write the default values to the settings file. */
	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------
	bool fullscreen		= settings.loadSetBool("game", "fullscreen", false);
	bool vsync			= settings.loadSetBool("game", "vsync", false);
	int target_fps		= settings.loadSetInt("game", "target_fps", 2000);
	if (!fullscreen) {
		// Only set the window size if the game is not fullscreen
		int window_width  = settings.loadSetInt("game", "window_width", 800);
		int window_height = settings.loadSetInt("game", "window_height", 600);
		ofSetWindowShape(window_width, window_height);
	}

	// Will still set the target fps if we are in vsync since it might be turned off later.
	ofSetFrameRate(target_fps);
	ofSetFullscreen(fullscreen);
	ofSetVerticalSync(vsync);
	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------
	
	// Load bindings
	vector<string> aliases = settings.getAllValues("alias", "alias");
	for (string& alias : aliases) {
		vector<string> bindings = settings.getAllValues("bindings", alias);
		for (string& binding : bindings) {
			engine->getKeyboardInput()->registerAlias(alias, binding);
		}
	}

	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------

	engine->setup();
	engine->PushState(MenuState::Instance());
}

//--------------------------------------------------------------
void ofApp::update(){
	ofxTimer timer("ofApp::update()");
	engine->update();
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofxTimer timer("ofApp::draw()");
	// Background
	ofBackground(ofColor::dimGrey);

	// Debug Values
	ofSetColor(ofColor::white);
	debugPush("FrameRate: " + ofToString(ofGetFrameRate()));
	//debugPush("WindowPosition: " + ofToString(ofVec2f(ofGetWindowPositionX(), ofGetWindowPositionY())));
	//debugPush("WindowSize: " + ofToString(ofGetWindowSize()));

	// GameState draw
	engine->draw();

	// Draw the debug info to screen
	debugDraw();

	//Reset inputs
	engine->getKeyboardInput()->resetPollingMaps();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	engine->getKeyboardInput()->keyPressed(key);
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	engine->getKeyboardInput()->keyReleased(key);
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
	engine->getKeyboardInput()->mousePressed(x, y, button);
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
	engine->getKeyboardInput()->mouseReleased(x, y, button);
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
