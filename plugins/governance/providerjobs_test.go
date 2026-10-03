package governance

import (
	"context"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	configstoreTables "github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newProviderJobTestPlugin is newAccessTestPlugin over the MCP-stamping VK (id vk-mcp-stamp,
// value mcpTestVKValue, every openai model), backed by a real SQLite config store that holds the
// provider job rows the ownership checks read.
func newProviderJobTestPlugin(t *testing.T) (*GovernancePlugin, configstore.ConfigStore) {
	t.Helper()
	ctx := context.Background()
	configStore, err := configstore.NewConfigStore(ctx, &configstore.Config{
		Enabled: true,
		Type:    configstore.ConfigStoreTypeSQLite,
		Config:  &configstore.SQLiteConfig{Path: t.TempDir() + "/providerjobs.db"},
	}, NewMockLogger())
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, configStore.Close(ctx)) })

	plugin := newAccessTestPlugin(t, buildVKForMCPStamping(nil), nil)
	plugin.configStore = configStore
	return plugin, configStore
}

func seedProviderJob(t *testing.T, store configstore.ConfigStore, provider, jobID string, ownerVK *string) {
	t.Helper()
	require.NoError(t, store.UpsertProviderJob(context.Background(), &configstoreTables.TableProviderJob{
		ID:               configstoreTables.ProviderJobID(configstoreTables.ProviderJobKindBatch, provider, jobID),
		Kind:             configstoreTables.ProviderJobKindBatch,
		Provider:         provider,
		JobID:            jobID,
		AccountingStatus: configstoreTables.ProviderJobAccountingStatusPending,
		VirtualKeyID:     ownerVK,
	}))
}

func batchRequest(requestType schemas.RequestType, batchID string) *schemas.BifrostRequest {
	req := &schemas.BifrostRequest{RequestType: requestType}
	switch requestType {
	case schemas.BatchRetrieveRequest:
		req.BatchRetrieveRequest = &schemas.BifrostBatchRetrieveRequest{Provider: schemas.OpenAI, BatchID: batchID}
	case schemas.BatchCancelRequest:
		req.BatchCancelRequest = &schemas.BifrostBatchCancelRequest{Provider: schemas.OpenAI, BatchID: batchID}
	case schemas.BatchResultsRequest:
		req.BatchResultsRequest = &schemas.BifrostBatchResultsRequest{Provider: schemas.OpenAI, BatchID: batchID}
	}
	return req
}

// A batch addressed by id is reachable through the virtual key that created it. A row naming another
// key is refused as not found; a row naming no key, no row at all, and a request that presented no
// key are all unrestricted, as they were.
func TestPreLLMHookBindsBatchesToTheCreatingVirtualKey(t *testing.T) {
	plugin, store := newProviderJobTestPlugin(t)
	own := "vk-mcp-stamp"
	other := "vk-someone-else"
	seedProviderJob(t, store, "openai", "batch-own", &own)
	seedProviderJob(t, store, "openai", "batch-other", &other)
	seedProviderJob(t, store, "openai", "batch-unowned", nil)

	for _, requestType := range []schemas.RequestType{schemas.BatchRetrieveRequest, schemas.BatchCancelRequest, schemas.BatchResultsRequest} {
		t.Run(string(requestType), func(t *testing.T) {
			_, shortCircuit, err := plugin.PreLLMHook(presentCtx(mcpTestVKValue), batchRequest(requestType, "batch-own"))
			require.NoError(t, err)
			assert.Nil(t, shortCircuit, "the creating key reaches its own batch")

			_, shortCircuit, err = plugin.PreLLMHook(presentCtx(mcpTestVKValue), batchRequest(requestType, "batch-other"))
			require.NoError(t, err)
			require.NotNil(t, shortCircuit, "another key's batch is refused")
			require.NotNil(t, shortCircuit.Error.StatusCode)
			assert.Equal(t, 404, *shortCircuit.Error.StatusCode, "refused as not found, not as forbidden")
			assert.Contains(t, shortCircuit.Error.Error.Message, "batch-other")

			_, shortCircuit, err = plugin.PreLLMHook(presentCtx(mcpTestVKValue), batchRequest(requestType, "batch-unowned"))
			require.NoError(t, err)
			assert.Nil(t, shortCircuit, "a row with no recorded key binds to nobody")

			_, shortCircuit, err = plugin.PreLLMHook(presentCtx(mcpTestVKValue), batchRequest(requestType, "batch-unknown"))
			require.NoError(t, err)
			assert.Nil(t, shortCircuit, "a batch with no row cannot be bound and stays reachable")

			_, shortCircuit, err = plugin.PreLLMHook(emptyCtx(), batchRequest(requestType, "batch-other"))
			require.NoError(t, err)
			assert.Nil(t, shortCircuit, "a request that presented no key is unrestricted")
		})
	}
}

// A batch list is narrowed to the batches the virtual key may see: another key's batches are
// dropped, its own and unbound ones stay, and the provider's pagination cursors are untouched.
func TestPostLLMHookFiltersBatchListToTheVirtualKey(t *testing.T) {
	plugin, store := newProviderJobTestPlugin(t)
	own := "vk-mcp-stamp"
	other := "vk-someone-else"
	seedProviderJob(t, store, "openai", "batch-own", &own)
	seedProviderJob(t, store, "openai", "batch-other", &other)
	seedProviderJob(t, store, "openai", "batch-unowned", nil)

	newList := func() *schemas.BifrostResponse {
		return &schemas.BifrostResponse{
			BatchListResponse: &schemas.BifrostBatchListResponse{
				Object:      "list",
				Data:        []schemas.BifrostBatchRetrieveResponse{{ID: "batch-own"}, {ID: "batch-other"}, {ID: "batch-unowned"}, {ID: "batch-unknown"}},
				HasMore:     true,
				LastID:      schemas.Ptr("batch-unknown"),
				ExtraFields: schemas.BifrostResponseExtraFields{RequestType: schemas.BatchListRequest, Provider: schemas.OpenAI},
			},
		}
	}
	ids := func(list *schemas.BifrostBatchListResponse) []string {
		out := make([]string, 0, len(list.Data))
		for _, item := range list.Data {
			out = append(out, item.ID)
		}
		return out
	}

	// The list request is evaluated first, which is what stamps the key on the context.
	ctx := presentCtx(mcpTestVKValue)
	_, shortCircuit, err := plugin.PreLLMHook(ctx, &schemas.BifrostRequest{RequestType: schemas.BatchListRequest, BatchListRequest: &schemas.BifrostBatchListRequest{Provider: schemas.OpenAI}})
	require.NoError(t, err)
	require.Nil(t, shortCircuit)
	result, _, err := plugin.PostLLMHook(ctx, newList(), nil)
	require.NoError(t, err)
	assert.Equal(t, []string{"batch-own", "batch-unowned", "batch-unknown"}, ids(result.BatchListResponse))
	assert.True(t, result.BatchListResponse.HasMore, "pagination is the provider's and is left alone")
	assert.Equal(t, "batch-unknown", *result.BatchListResponse.LastID)

	// With no key presented the list is the provider's answer, unchanged.
	anonymous := emptyCtx()
	_, shortCircuit, err = plugin.PreLLMHook(anonymous, &schemas.BifrostRequest{RequestType: schemas.BatchListRequest, BatchListRequest: &schemas.BifrostBatchListRequest{Provider: schemas.OpenAI}})
	require.NoError(t, err)
	require.Nil(t, shortCircuit)
	result, _, err = plugin.PostLLMHook(anonymous, newList(), nil)
	require.NoError(t, err)
	assert.Equal(t, []string{"batch-own", "batch-other", "batch-unowned", "batch-unknown"}, ids(result.BatchListResponse))
}
