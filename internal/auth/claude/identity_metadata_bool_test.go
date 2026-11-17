package claude

import (
	"sync"
	"testing"
)

func TestReadMetadataBoolConcurrentUpdates(t *testing.T) {
	if ReadMetadataBool(nil, "setup_token") {
		t.Fatal("nil metadata must not indicate a setup token")
	}
	var metadata map[string]any
	if ReadMetadataBool(&metadata, "setup_token") {
		t.Fatal("empty metadata must not indicate a setup token")
	}
	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 1000; i++ {
			StoreMetadataValue(&metadata, "setup_token", i%2 == 0)
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 1000; i++ {
			_ = ReadMetadataBool(&metadata, "setup_token")
		}
	}()
	close(start)
	wg.Wait()
	StoreMetadataValue(&metadata, "setup_token", true)
	if !ReadMetadataBool(&metadata, "setup_token") {
		t.Fatal("stored setup token flag was lost")
	}
	StoreMetadataValue(&metadata, "setup_token", "true")
	if ReadMetadataBool(&metadata, "setup_token") {
		t.Fatal("non-boolean metadata must not indicate a setup token")
	}
}
