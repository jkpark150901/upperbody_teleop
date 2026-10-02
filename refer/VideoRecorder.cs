using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System;
#if UNITY_EDITOR
using UnityEditor.Recorder;
using UnityEditor.Recorder.Input;
#endif

public class VideoRecorder : MonoBehaviour
{
    // 단일 카메라 할당으로 변경
    [SerializeField] private Camera targetCamera;
    private string videoFolderPath;

#if UNITY_EDITOR
    // 단일 컨트롤러 및 렌더텍스처로 변경
    private RecorderController recorderController;
    private RenderTexture createdRenderTexture;
#endif

    void Start()
    {
        if (!ValidateComponents()) return;

        // Assets의 상위 폴더(Project Root)에 RecordedVideos 폴더 경로 설정
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        videoFolderPath = Path.Combine(projectRoot, "RecordedVideos");

        try
        {
            if (!Directory.Exists(videoFolderPath))
            {
                Directory.CreateDirectory(videoFolderPath);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"폴더 생성 실패: {e.Message}");
            return;
        }

#if UNITY_EDITOR
        InitializeRecorders();
        StartVideoRecording();
#endif
    }

    private bool ValidateComponents()
    {
        // 단일 카메라 체크로 변경
        if (targetCamera == null)
        {
            Debug.LogError("VideoRecorder: 카메라가 할당되지 않았습니다.");
            return false;
        }
        return true;
    }

#if UNITY_EDITOR
    private void InitializeRecorders()
    {
        // 현재 날짜 및 시간을 파일명 형식으로 추출
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

        try
        {
            var settings = ScriptableObject.CreateInstance<RecorderControllerSettings>();
            var movieRecorder = ScriptableObject.CreateInstance<MovieRecorderSettings>();
            
            movieRecorder.Enabled = true;
            
            // 파일명 설정: 단일 카메라용 이름
            string fileName = $"{timestamp}_Camera";
            movieRecorder.OutputFile = Path.Combine(videoFolderPath, fileName);

            RenderTexture renderTexture = targetCamera.targetTexture;
            if (renderTexture == null)
            {
                renderTexture = new RenderTexture(1920, 1080, 24, RenderTextureFormat.ARGBHalf);
                renderTexture.Create();
                createdRenderTexture = renderTexture;
                targetCamera.targetTexture = renderTexture;
            }

            movieRecorder.ImageInputSettings = new RenderTextureInputSettings
            {
                OutputWidth = 1920,
                OutputHeight = 1080,
                RenderTexture = renderTexture
            };

            settings.AddRecorderSettings(movieRecorder);
            settings.FrameRate = 30;
            
            recorderController = new RecorderController(settings);
        }
        catch (Exception e)
        {
            Debug.LogError($"레코더 초기화 실패: {e.Message}");
        }
    }

    private void StartVideoRecording()
    {
        if (recorderController != null && !recorderController.IsRecording())
        {
            recorderController.PrepareRecording();
            recorderController.StartRecording();
        }
        Debug.Log($"녹화 시작. 저장 위치: {videoFolderPath}");
    }

    private void StopVideoRecording()
    {
        if (recorderController != null && recorderController.IsRecording())
        {
            recorderController.StopRecording();
        }
        Debug.Log("녹화 종료 및 저장 완료.");
    }
#endif

    void OnDestroy()
    {
#if UNITY_EDITOR
        StopVideoRecording();

        if (createdRenderTexture != null)
        {
            if (createdRenderTexture.IsCreated()) createdRenderTexture.Release();
            DestroyImmediate(createdRenderTexture);
        }
        recorderController = null;
#endif
    }
}