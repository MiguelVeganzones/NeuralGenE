#pragma comment(lib, "odbc32.lib")
#pragma comment(lib, "odbcbcp.lib")

//#include <iostream>
//#include <stdio.h>
//#include <windows.h>
//#include <sql.h>  
//#include <sqlext.h>  
//#include <odbcss.h>  
//#include <sstream>

#include <iostream>
#include <windows.h>
#include <sql.h>
#include <sqlext.h>
#include <odbcss.h>
#include <stdio.h>
#include <string.h>

SQLHENV  henv  = SQL_NULL_HENV;
HDBC     hdbc1 = SQL_NULL_HDBC, hdbc2 = SQL_NULL_HDBC;
SQLHSTMT hstmt2 = SQL_NULL_HSTMT;

void Cleanup()
{
    if (hstmt2 != SQL_NULL_HSTMT)
        SQLFreeHandle(SQL_HANDLE_STMT, hstmt2);

    if (hdbc1 != SQL_NULL_HDBC)
    {
        SQLDisconnect(hdbc1);
        SQLFreeHandle(SQL_HANDLE_DBC, hdbc1);
    }

    if (hdbc2 != SQL_NULL_HDBC)
    {
        SQLDisconnect(hdbc2);
        SQLFreeHandle(SQL_HANDLE_DBC, hdbc2);
    }

    if (henv != SQL_NULL_HENV)
        SQLFreeHandle(SQL_HANDLE_ENV, henv);
}

int main()
{
    SQLCHAR retConString[1024];
    SQLCHAR SQLState[1024];
    SQLCHAR message[1024];
    RETCODE retcode;

    // BCP variables.
    char terminator = '\0';

    // bcp_done takes a different format return code because it returns number of rows bulk copied
    // after the last bcp_batch call.
    DBINT cRowsDone = 0;

    // Set up separate return code for bcp_sendrow so it is not using the same retcode as SQLFetch.
    RETCODE SendRet;

    // Column variables.  cbCola and cbColb must be defined right before Cola and szColb because
    // they are used as bulk copy indicator variables.
    struct ColaData
    {
        int        cbCola;
        SQLINTEGER Cola;
    } ColaInst;

    struct ColbData
    {
        int     cbColb;
        SQLCHAR szColb[11];
    } ColbInst;

    // Allocate the ODBC environment and save handle.
    retcode = SQLAllocHandle(SQL_HANDLE_ENV, NULL, &henv);
    if ((retcode != SQL_SUCCESS_WITH_INFO) && (retcode != SQL_SUCCESS))
    {
        printf("SQLAllocHandle(Env) Failed\n\n");
        Cleanup();
        return (9);
    }

    // Notify ODBC that this is an ODBC 3.0 app.
    retcode = SQLSetEnvAttr(henv, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)SQL_OV_ODBC3, SQL_IS_INTEGER);
    if ((retcode != SQL_SUCCESS_WITH_INFO) && (retcode != SQL_SUCCESS))
    {
        printf("SQLSetEnvAttr(ODBC version) Failed\n\n");
        Cleanup();
        return (9);
    }

    // Allocate ODBC connection handle, set bulk copy mode, and connect.
    retcode = SQLAllocHandle(SQL_HANDLE_DBC, henv, &hdbc1);
    if ((retcode != SQL_SUCCESS_WITH_INFO) && (retcode != SQL_SUCCESS))
    {
        printf("SQLAllocHandle(hdbc1) Failed\n\n");
        Cleanup();
        return (9);
    }

    retcode = SQLSetConnectAttr(hdbc1, SQL_COPT_SS_BCP, (void*)SQL_BCP_ON, SQL_IS_INTEGER);
    if ((retcode != SQL_SUCCESS_WITH_INFO) && (retcode != SQL_SUCCESS))
    {
        printf("SQLSetConnectAttr(hdbc1) Failed\n\n");
        Cleanup();
        return (9);
    }

    retcode = SQLConnect(hdbc1, (UCHAR*)"SQL Server", SQL_NTS, (UCHAR*)"Test_login", SQL_NTS, (UCHAR*)"1234", SQL_NTS);
    if ((retcode != SQL_SUCCESS) && (retcode != SQL_SUCCESS_WITH_INFO))
    {
        retcode = SQLGetDiagRec(SQL_HANDLE_DBC, hdbc1, 1, SQLState, NULL, message, 1024, NULL);
        std::cout << '\n' << retcode << std::endl;
        std::cout << SQLState << std::endl;
        std::cout << message << std::endl;
        printf("SQLConnect(hbdc1) Failed\n\n");
        Cleanup();
        return (9);
    }

    // Initialize the bulk copy.
    retcode = bcp_init(hdbc1, "AdventureWorks..BCPTarget", NULL, NULL, DB_IN);
    if ((retcode != SUCCEED))
    {
        retcode = SQLGetDiagRec(SQL_HANDLE_DBC, hdbc1, 1, SQLState, NULL, message, 1024, NULL);
        std::cout << '\n' << retcode << std::endl;
        std::cout << SQLState << std::endl;
        std::cout << message << std::endl;
        printf("bcp_init(hdbc1) Failed\n\n");
        Cleanup();
        return (9);
    }

    // Bind the program variables for the bulk copy.
    retcode = bcp_bind(hdbc1, (BYTE*)&ColaInst.cbCola, 4, SQL_VARLEN_DATA, NULL, (INT)NULL, SQLINT4, 1);
    if ((retcode != SUCCEED))
    {
        printf("bcp_bind(hdbc1) Failed\n\n");
        Cleanup();
        return (9);
    }

    // Could normally use strlen to calculate the bcp_bind cbTerm parameter, but this terminator
    // is a null byte (\0), which gives strlen a value of 0. Explicitly give cbTerm a value of 1.
    retcode = bcp_bind(hdbc1, (BYTE*)&ColbInst.cbColb, 4, 11, (UCHAR*)terminator, 1, SQLCHARACTER, 2);
    if ((retcode != SUCCEED))
    {
        printf("bcp_bind(hdbc1) Failed\n\n");
        Cleanup();
        return (9);
    }

    // Allocate second ODBC connection handle so bulk copy and cursor operations do not conflict.
    retcode = SQLAllocHandle(SQL_HANDLE_DBC, henv, &hdbc2);
    if ((retcode != SQL_SUCCESS_WITH_INFO) && (retcode != SQL_SUCCESS))
    {
        printf("SQLAllocHandle(hdbc2) Failed\n\n");
        Cleanup();
        return (9);
    }

    // Sample uses Integrated Security, create SQL Server DSN using Windows NT authentication.
    retcode = SQLConnect(hdbc2, (UCHAR*)"AdventureWorks", SQL_NTS, (UCHAR*)"", SQL_NTS, (UCHAR*)"", SQL_NTS);
    if ((retcode != SQL_SUCCESS) && (retcode != SQL_SUCCESS_WITH_INFO))
    {
        printf("SQLConnect(hbdc2) Failed\n\n");
        Cleanup();
        return (9);
    }

    // Allocate ODBC statement handle.
    retcode = SQLAllocHandle(SQL_HANDLE_STMT, hdbc2, &hstmt2);
    if ((retcode != SQL_SUCCESS_WITH_INFO) && (retcode != SQL_SUCCESS))
    {
        printf("SQLAllocHandle(hstmt2) Failed\n\n");
        Cleanup();
        return (9);
    }

    SQLLEN lDataLengthA;
    SQLLEN lDataLengthB;

    // Bind the SELECT statement to the same program variables bound to the bulk copy operation.
    retcode = SQLBindCol(hstmt2, 1, SQL_C_SLONG, &ColaInst.Cola, 0, &lDataLengthA);
    if ((retcode != SQL_SUCCESS_WITH_INFO) && (retcode != SQL_SUCCESS))
    {
        printf("SQLBindCol(hstmt2) Failed\n\n");
        Cleanup();
        return (9);
    }

    retcode = SQLBindCol(hstmt2, 2, SQL_C_CHAR, &ColbInst.szColb, 11, &lDataLengthB);
    if ((retcode != SQL_SUCCESS_WITH_INFO) && (retcode != SQL_SUCCESS))
    {
        printf("SQLBindCol(hstmt2) Failed\n\n");
        Cleanup();
        return (9);
    }

    // Execute SELECT statement to build a cursor containing data to be bulk copied to new table.
    retcode = SQLExecDirect(hstmt2, (UCHAR*)"SELECT * FROM BCPSource", SQL_NTS);
    if ((retcode != SQL_SUCCESS) && (retcode != SQL_SUCCESS_WITH_INFO))
    {
        printf("SQLExecDirect Failed\n\n");
        Cleanup();
        return (9);
    }
    // Go into a loop fetching rows from the cursor until each row is fetched. Because the
    // bcp_bind calls and SQLBindCol calls each reference the same variables, each fetch fills
    // the variables used by bcp_sendrow, so all you have to do to send the data to SQL Server is
    // to call bcp_sendrow.
    while ((retcode = SQLFetch(hstmt2)) != SQL_NO_DATA)
    {
        if ((retcode != SQL_SUCCESS) && (retcode != SQL_SUCCESS_WITH_INFO))
        {
            printf("SQLFetch Failed\n\n");
            Cleanup();
            return (9);
        }

        ColaInst.cbCola = (int)lDataLengthA;
        ColbInst.cbColb = (int)lDataLengthB;

        if ((SendRet = bcp_sendrow(hdbc1)) != SUCCEED)
        {
            printf("bcp_sendrow(hdbc1) Failed\n\n");
            Cleanup();
            return (9);
        }
    }

    // Signal the end of the bulk copy operation.
    cRowsDone = bcp_done(hdbc1);
    if ((cRowsDone == -1))
    {
        printf("bcp_done(hdbc1) Failed\n\n");
        Cleanup();
        return (9);
    }

    printf("Number of rows bulk copied after last bcp_batch call = %d.\n", cRowsDone);

    // Cleanup.
    SQLFreeHandle(SQL_HANDLE_STMT, hstmt2);
    SQLDisconnect(hdbc1);
    SQLFreeHandle(SQL_HANDLE_DBC, hdbc1);
    SQLDisconnect(hdbc2);
    SQLFreeHandle(SQL_HANDLE_DBC, hdbc2);
    SQLFreeHandle(SQL_HANDLE_ENV, henv);
}

//int main()
//{
//    SQLHENV   SQLEnvHandle        = NULL;
//    SQLHDBC   SQLConnectionHandle = NULL;
//    SQLHDBC   hdbc2               = NULL;
//    SQLHSTMT  SQLStatementHandle  = NULL;
//    SQLRETURN retCode             = 0;
//
//    char    SQLQuery[] = "SELECT idx, generation, input, ground_truth, prediction FROM Inference";
//    SQLCHAR retConString[1024];
//    SQLCHAR SQLState[1024];
//    SQLCHAR message[1024];
//
//
//    auto ret = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &SQLEnvHandle);
//    std::cout << ret << '\n';
//
//    ret = SQLSetEnvAttr(SQLEnvHandle, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)SQL_OV_ODBC3, 0);
//    std::cout << ret << '\n';
//
//    ret = SQLAllocHandle(SQL_HANDLE_DBC, SQLEnvHandle, &SQLConnectionHandle);
//    std::cout << ret << '\n';
//
//    ret = SQLSetConnectAttr(SQLConnectionHandle, SQL_LOGIN_TIMEOUT, (SQLPOINTER)5, 0);
//    std::cout << ret << '\n';
//
//    SQLSetConnectAttr(SQLConnectionHandle, SQL_COPT_SS_BCP, (void*)SQL_BCP_ON, SQL_IS_INTEGER);
//
//    ret = SQLDriverConnect(SQLConnectionHandle,
//                           NULL,
//                           (SQLCHAR*)"DRIVER={ODBC Driver 17 for SQL Server}; "
//                                     "SERVER=(localdb)\\MSSQLLocalDB; "
//                                     "DATABASE=Training_data;",
//                           SQL_NTS,
//                           retConString,
//                           1024,
//                           NULL,
//                           SQL_DRIVER_NOPROMPT);
//
//    std::cout << retConString << std::endl;
//
//    switch (ret)
//    {
//    case SQL_SUCCESS:
//        std::cout << "here0";
//        break;
//    case SQL_NO_DATA_FOUND:
//        std::cout << "here1";
//        break;
//    case SQL_ERROR:
//        std::cout << "here2";
//        break;
//    case SQL_INVALID_HANDLE:
//        std::cout << "here3";
//        break;
//    default:
//        std::cout << ret << std::endl;
//    }
//
//    float input, ground_truth, pred;
//    int   idx, gen;
//
//    int   gen_[5]          = { 3, 3, 3, 3, 3 };
//    float input_[5]        = { 0, 1, 2, 3, 4 };
//    float ground_truth_[5] = { 0, 1, 2, 3, 4 };
//    float pred_[5]         = { 0.2f, 1.2f, 2.2f, 3.2f, 4.2f };
//
//    ret = bcp_init(SQLConnectionHandle, "Inference", NULL, NULL, DB_IN);
//    std::cout << ret << '\n';
//
//    ret = bcp_bind(SQLConnectionHandle, (BYTE*)gen_, 4, 5, NULL, (INT)NULL, SQL_INTEGER, 2);
//    std::cout << ret << '\n';
//
//    ret = bcp_bind(SQLConnectionHandle, (BYTE*)input_, 4, 5, NULL, (INT)NULL, SQL_REAL, 3);
//    std::cout << ret << '\n';
//
//    ret = bcp_bind(SQLConnectionHandle, (BYTE*)ground_truth_, 4, 5, NULL, (INT)NULL, SQL_REAL, 4);
//    std::cout << ret << '\n';
//
//    ret = bcp_bind(SQLConnectionHandle, (BYTE*)pred_, 4, 5, NULL, (INT)NULL, SQL_REAL, 5);
//    std::cout << ret << '\n';
//
//    ret = SQLAllocHandle(SQL_HANDLE_DBC, SQLEnvHandle, &hdbc2);
//    std::cout << ret << '\n';
//
//    ret = SQLDriverConnect(hdbc2,
//                           NULL,
//                           (SQLCHAR*)"DRIVER={ODBC Driver 17 for SQL Server}; "
//                                     "SERVER=(localdb)\\MSSQLLocalDB; "
//                                     "DATABASE=Training_data;",
//                           SQL_NTS,
//                           retConString,
//                           1024,
//                           NULL,
//                           SQL_DRIVER_NOPROMPT);
//    std::cout << ret << '\n';
//
//    ret = SQLAllocHandle(SQL_HANDLE_STMT, hdbc2, &SQLStatementHandle);
//    std::cout << ret << '\n';
//
//    ret = SQLBindCol(SQLStatementHandle, 2, SQL_INTEGER, gen_, 5, NULL);
//    std::cout << ret << '\n';
//
//    ret = SQLBindCol(SQLStatementHandle, 3, SQL_REAL, input_, 5, NULL);
//    std::cout << ret << '\n';
//
//    ret = SQLBindCol(SQLStatementHandle, 4, SQL_REAL, ground_truth_, 5, NULL);
//    std::cout << ret << '\n';
//
//    ret = SQLBindCol(SQLStatementHandle, 5, SQL_REAL, pred_, 5, NULL);
//    std::cout << ret << '\n';
//
//    ret = SQLExecDirect(SQLStatementHandle, (SQLCHAR*)SQLQuery, SQL_NTS);
//    std::cout << ret << '\n';
//
//    while (SQLFetch(SQLStatementHandle) == SQL_SUCCESS)
//    {
//        bcp_sendrow(SQLConnectionHandle);
//    }
//
//    // bcp_sendrow();
//    // bcp_batch(); // At least once per 1000 rows
//    auto rows = bcp_done(SQLConnectionHandle);
//    std::cout << "rows: " << rows << std::endl;
//
//    ret = SQLGetDiagRec(SQL_HANDLE_DBC, SQLConnectionHandle, 1, SQLState, NULL, message, 1024, NULL);
//    std::cout << '\n' << ret << std::endl;
//    std::cout << SQLState << std::endl;
//    std::cout << message << std::endl;
//
//    // ret = SQLGetDiagRec(SQL_HANDLE_DBC, SQLConnectionHandle, 1, SQLState, NULL, message, 1024, NULL);
//    // std::cout << '\n' << ret << std::endl;
//    // std::cout << SQLState << std::endl;
//    // std::cout << message << std::endl;
//    //// std::cout << retConString << std::endl;
//
//    //// Allocate statement handle
//    // ret = SQLAllocHandle(SQL_HANDLE_STMT, SQLConnectionHandle, &SQLStatementHandle);
//    // std::cout << ret << std::endl;
//
//    // ret = SQLExecDirect(SQLStatementHandle, (SQLCHAR*)SQLQuery, SQL_NTS);
//    // std::cout << ret << std::endl;
//    // std::cout << "----------------\n " << std::endl;
//
//    // float input, ground_truth, pred;
//    // int   idx, gen;
//
//    // while (SQLFetch(SQLStatementHandle) == SQL_SUCCESS)
//    //{
//    //     SQLGetData(SQLStatementHandle, 1, SQL_INTEGER, &idx, sizeof(idx), NULL);
//    //     SQLGetData(SQLStatementHandle, 2, SQL_INTEGER, &gen, sizeof(gen), NULL);
//    //     SQLGetData(SQLStatementHandle, 3, SQL_REAL, &input, sizeof(input), NULL);
//    //     SQLGetData(SQLStatementHandle, 4, SQL_REAL, &ground_truth, sizeof(ground_truth), NULL);
//    //     SQLGetData(SQLStatementHandle, 5, SQL_REAL, &pred, sizeof(pred), NULL);
//    //     std::cout << idx << ' ' << gen << ' ' << input << ' ' << ground_truth << ' ' << pred << std::endl;
//    // }
//
//    // int   gen_[5]         = { 3, 3, 3, 3, 3 };
//    // float input_[5]       = { 0, 1, 2, 3, 4 };
//    // float ground_truth_[5] = { 0, 1, 2, 3, 4 };
//    // float pred_[5]        = { 0.2f, 1.2f, 2.2f, 3.2f, 4.2f };
//
//    // ret = SQLSetStmtAttr(SQLStatementHandle, SQL_ATTR_ROW_ARRAY_SIZE, (SQLPOINTER)5, 0);
//    // std::cout << ret << std::endl;
//    // SQLBindCol(SQLStatementHandle, 2, SQL_INTEGER, gen_, 5, NULL);
//    // SQLBindCol(SQLStatementHandle, 3, SQL_REAL, input_, 5, NULL);
//    // SQLBindCol(SQLStatementHandle, 4, SQL_REAL, ground_truth_, 5, NULL);
//    // SQLBindCol(SQLStatementHandle, 5, SQL_REAL, pred_, 5, NULL);
//
//    // SQLFetchScroll(SQLStatementHandle, SQL_FETCH_LAST, 0);
//
//    // SQLBulkOperations(SQLStatementHandle, SQL_ADD);
//
//    // ret = SQLGetDiagRec(SQL_HANDLE_STMT, SQLStatementHandle, 1, SQLState, NULL, message, 1024, NULL);
//    // std::cout << '\n' << ret << std::endl;
//    // std::cout << SQLState << std::endl;
//    // std::cout << message << std::endl;
//
//    SQLFreeHandle(SQL_HANDLE_STMT, SQLStatementHandle);
//    SQLDisconnect(SQLConnectionHandle);
//    SQLFreeHandle(SQL_HANDLE_DBC, SQLConnectionHandle);
//    SQLFreeHandle(SQL_HANDLE_ENV, SQLEnvHandle);
//    return EXIT_SUCCESS;
//}
