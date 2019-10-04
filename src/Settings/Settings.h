#pragma once
#include "ofMain.h"
#include "ofxIni/ofxIniFile.h"

/*
This library simply wraps around some of the ofxIni.h functions. It allows you to read from an ini file and 
write a default value if the key pair does not exist
*/

namespace Settings {
	bool loadSetBool(ofxIniFile& file, string section, string key, bool defaultValue) {
		bool result = file.getBool(section, key, defaultValue);
		file.setBool(section, key, result);
		file.save();
		return result;
	}

	int loadSetInt(ofxIniFile& file, string section, string key, int defaultValue) {
		int result = file.getInt(section, key, defaultValue);
		file.setInt(section, key, result);
		file.save();
		return result;
	}

	float loadSetFloat(ofxIniFile& file, string section, string key, float defaultValue) {
		float result = file.getFloat(section, key, defaultValue);
		file.setFloat(section, key, result);
		file.save();
		return result;
	}

	long loadSetLong(ofxIniFile& file, string section, string key, long defaultValue) {
		long result = file.getLong(section, key, defaultValue);
		file.setLong(section, key, result);
		file.save();
		return result;
	}

	string loadSetString(ofxIniFile& file, string section, string key, string defaultValue) {
		string& result = file.getString(section, key, defaultValue);
		file.setString(section, key, result);
		file.save();
		return result;
	}

	ofVec3f loadSetVec3f(ofxIniFile& file, string section, string key, string defaultValue) {
		IniVec3f& result = file.getVec3f(section, key, defaultValue);
		file.setVec3f(section, key, result);
		file.save();
		return *(ofVec3f *)&result;
	}
}