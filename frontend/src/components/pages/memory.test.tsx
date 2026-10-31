import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { memorySimilarityThreshold } from "@/lib/memory-threshold"

const { api, success } = vi.hoisted(() => ({ api: vi.fn(), success: vi.fn() }))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: api }))
vi.mock("@/components/ui/sonner", () => ({ toast: { success } }))
vi.mock("@/lib/utils", async original => ({
  ...(await original<typeof import("@/lib/utils")>()), getApiUrl: () => "http://synthetic.local",
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key, tDynamic: (key: string) => key, locale: "en" }),
}))

import { MemoryPage } from "./memory"

const response = (body: unknown, ok = true) => ({ ok, json: async () => body })
let threshold: unknown
let save: () => Promise<unknown>

beforeEach(() => {
  api.mockReset()
  success.mockReset()
  threshold = 0.4
  save = async () => response({ memory_id: "synthetic-note" })
  api.mockImplementation((url: string, options?: RequestInit) => {
    if (options?.method === "POST") return save()
    if (url.endsWith("/store-info")) return Promise.resolve(response({ similarity_threshold: threshold }))
    if (url.includes("/list?")) return Promise.resolve(response({ memories: [], total_count: 0, filters_used: {} }))
    if (url.endsWith("/stats")) return Promise.resolve(response({ total_count: 0, category_counts: {}, tag_counts: {}, memory_store_type: "synthetic" }))
    throw new Error("Unexpected synthetic API request")
  })
})
afterEach(cleanup)

describe("memory report regressions", () => {
  it.each([[undefined, 0.4], [NaN, 0.4], [Infinity, 0.4], ["1.5", 0.4], [-1, 0.1], [4, 2], [0.25, 0.25]])(
    "bounds the display threshold %s to %s", (input, expected) => {
      expect(memorySimilarityThreshold(input)).toBe(expected)
    },
  )

  it("uses the bounded server threshold and omits an unproduced task category", async () => {
    threshold = 0.25
    render(<MemoryPage />)
    await waitFor(() => expect(api.mock.calls.some(([url]) => String(url).includes("similarity_threshold=0.25"))).toBe(true))
    expect(screen.queryByText("memory.filters.categoryOptions.execution_memory")).toBeNull()
  })

  it("submits once while pending, then closes and acknowledges the successful save", async () => {
    let finish!: (value: unknown) => void
    save = () => new Promise(resolve => { finish = resolve })
    render(<MemoryPage />)
    await screen.findByText("memory.empty.title")
    fireEvent.click(screen.getByRole("button", { name: /memory.header.create/ }))
    fireEvent.change(screen.getByLabelText("memory.form.contentLabel"), { target: { value: "Synthetic preference" } })
    const button = screen.getByRole("button", { name: "common.save" })
    fireEvent.click(button)
    fireEvent.click(button)
    expect(button).toBeDisabled()
    expect(api.mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(1)
    finish(response({ memory_id: "synthetic-note" }))
    await waitFor(() => expect(success).toHaveBeenCalledExactlyOnceWith("memory.success.created"))
    expect(screen.queryByRole("dialog")).toBeNull()
  })

  it("allows a retry after failure without reporting success prematurely", async () => {
    save = async () => response({}, false)
    render(<MemoryPage />)
    await screen.findByText("memory.empty.title")
    fireEvent.click(screen.getByRole("button", { name: /memory.header.create/ }))
    fireEvent.change(screen.getByLabelText("memory.form.contentLabel"), { target: { value: "Synthetic preference" } })
    fireEvent.click(screen.getByRole("button", { name: "common.save" }))
    await waitFor(() => expect(screen.getByRole("button", { name: "common.save" })).toBeEnabled())
    expect(success).not.toHaveBeenCalled()
    save = async () => response({ memory_id: "synthetic-note" })
    fireEvent.click(screen.getByRole("button", { name: "common.save" }))
    await waitFor(() => expect(success).toHaveBeenCalledExactlyOnceWith("memory.success.created"))
    expect(api.mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(2)
  })
})
