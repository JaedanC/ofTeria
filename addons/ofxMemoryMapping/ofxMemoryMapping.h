#pragma once
#ifndef OFXMEMORYMAPPING_H
#define OFXMEMORYMAPPING_H

#include "ofMain.h"

class ofxMemoryMapping
{
private:
	HANDLE windowsFileHandle;
	HANDLE windowsFileMapping;
	LPVOID windowsFileView;
	string filename;
	unsigned int fileSize;

	void close();
	void load(const string& filename, const size_t& fileSize);
	void windowsCloseFileHandle();
	void windowsCloseFileMappingHandle();
	void windowsCloseFileView();
	void windowsCreateFileHandle(const string& filename);
	void windowsCreateFileMappingHandle(const size_t& fileSize);
	void windowsCreateFileView();

public:
	ofxMemoryMapping();
	ofxMemoryMapping(const string& filename);
	~ofxMemoryMapping();

	void load(const string& filename);
	void write(const void* data, const size_t& offset, const size_t& bytes);
	void read(void* dataResult, const size_t& offset, const size_t& bytes) const;
	void resize(const size_t& newSize);
	unsigned int getFileSize() const;
	void info(const string& func) const;
};

#endif /* OFXMEMORYMAPPING_H */